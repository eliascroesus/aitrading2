"""FastAPI server: live websocket feed + REST control for the dashboard."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import Config
from .data import load_datasets
from .hive import Hive
from .instruments import get_instrument
from .propfirm import get_rules
from .store import Bus, Store

log = logging.getLogger("hive.server")
STATIC = Path(__file__).resolve().parent / "static"


class OrderIn(BaseModel):
    text: str


class DeployIn(BaseModel):
    uid: str


class LoginIn(BaseModel):
    password: str


LOGIN_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>HIVEMIND login</title>
<style>
body{margin:0;min-height:100vh;display:grid;place-items:center;background:#090b0e;color:#e7eaf0;font:14px system-ui,sans-serif;padding:16px}
form{box-sizing:border-box;background:#11151b;border:1px solid #2a3441;border-radius:14px;padding:28px;width:100%;max-width:340px}
h1{font-size:16px;letter-spacing:.25em;margin:0 0 18px;color:#f5b544}
input{width:100%;box-sizing:border-box;padding:11px 12px;border-radius:9px;border:1px solid #2a3441;background:#0d1014;color:#e7eaf0;font-size:14px}
button{margin-top:12px;width:100%;padding:11px;border:none;border-radius:9px;background:#f5b544;color:#1b1305;font-weight:600;font-size:14px;cursor:pointer}
p{color:#ff5d6c;min-height:18px;margin:10px 0 0;font-size:13px}
</style></head><body>
<form id="f"><h1>&#11042; HIVEMIND</h1><input id="p" type="password" placeholder="Dashboard password" autofocus>
<button>Enter</button><p id="e"></p></form>
<script>
document.getElementById('f').onsubmit=async(ev)=>{ev.preventDefault();
const r=await fetch('/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:document.getElementById('p').value})});
if(r.ok){location.href='/';}else{document.getElementById('e').textContent='Wrong password';}};
</script></body></html>"""


class SpawnIn(BaseModel):
    specialty: str
    mission: str
    focus: str = "any"


def create_app(cfg: Config) -> FastAPI:
    state: dict = {"hive": None, "boot": {"stage": "starting", "error": None, "log": []}, "bus": None}

    def boot_log(msg: str) -> None:
        state["boot"]["stage"] = msg
        state["boot"]["log"].append([time.time(), msg])
        log.info(msg)

    async def boot() -> None:
        try:
            rules = get_rules(cfg.prop_preset)
            instruments = {s: get_instrument(s, cfg.ratio_overrides) for s in cfg.instruments}
            store = Store(cfg.data_dir / "hive.db")
            bus = Bus(store)
            state["bus"] = bus
            boot_log(f"Loading market data ({cfg.resolved_data_source}) for {', '.join(instruments)} at {cfg.timeframes} min ...")
            datasets, files = await asyncio.to_thread(load_datasets, cfg, instruments)
            boot_log(f"Loaded {len(datasets)} datasets. Starting {cfg.workers} evaluation workers ...")
            hive = Hive(cfg, rules, instruments, datasets, files, store, bus)
            if cfg.alpaca_key and cfg.alpaca_secret:
                try:
                    from .broker import PaperTrader

                    hive.paper = PaperTrader(cfg, rules, instruments, bus, store)
                    await hive.paper.refresh_account()
                    hive.paper.resume_if_deployed()
                except Exception as exc:  # paper trading is optional
                    bus.emit("error", f"Alpaca paper trading unavailable: {exc}")
            await hive.start()
            state["hive"] = hive
            boot_log("Hive online.")
        except Exception as exc:
            log.exception("boot failed")
            state["boot"]["error"] = f"{type(exc).__name__}: {exc}"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(boot())
        yield
        task.cancel()
        hive: Hive | None = state["hive"]
        if hive:
            if hive.paper:
                hive.paper.task and hive.paper.task.cancel()
            await hive.stop()

    app = FastAPI(title="HIVEMIND", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    # optional password protection (set DASHBOARD_PASSWORD whenever the dashboard is reachable from the internet)
    password = cfg.dashboard_password
    token = hmac.new((password or "").encode(), b"hivemind-session-v1", hashlib.sha256).hexdigest()

    def authed(cookies) -> bool:
        return not password or hmac.compare_digest(cookies.get("hive_auth", ""), token)

    @app.middleware("http")
    async def require_login(request: Request, call_next):
        if password and request.url.path != "/login" and not authed(request.cookies):
            if request.url.path.startswith("/api/") or request.url.path.startswith("/static/"):
                return JSONResponse({"detail": "login required"}, status_code=401)
            return RedirectResponse("/login")
        return await call_next(request)

    @app.get("/login")
    async def login_page():
        if not password:
            return RedirectResponse("/")
        return HTMLResponse(LOGIN_PAGE)

    @app.post("/login")
    async def login(body: LoginIn, request: Request):
        if not password or not hmac.compare_digest(body.password, password):
            await asyncio.sleep(0.5)
            raise HTTPException(401, "wrong password")
        resp = JSONResponse({"ok": True})
        secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
        # SameSite=None over https so the login also works when the dashboard is embedded (e.g. a Hugging Face Space page)
        resp.set_cookie("hive_auth", token, max_age=60 * 60 * 24 * 30, httponly=True,
                        samesite="none" if secure else "lax", secure=secure)
        return resp

    def need_hive() -> Hive:
        hive = state["hive"]
        if hive is None:
            raise HTTPException(503, "hive is still booting")
        return hive

    def snapshot() -> dict:
        hive = state["hive"]
        if hive is None:
            return {"booting": True, "boot": state["boot"]}
        return hive.snapshot()

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/state")
    async def api_state():
        return JSONResponse(snapshot())

    @app.get("/api/strategy/{uid}")
    async def api_strategy(uid: str):
        rec = await need_hive().detail(uid)
        if rec is None:
            raise HTTPException(404, "unknown strategy")
        return rec

    @app.get("/api/strategy/{uid}/export")
    async def api_export(uid: str):
        hive = need_hive()
        rec = await hive.detail(uid)
        if rec is None:
            raise HTTPException(404, "unknown strategy")
        payload = {"exported": time.strftime("%Y-%m-%d %H:%M:%S"), "challenge": hive.rules.to_dict(), **rec}
        return JSONResponse(payload, headers={"Content-Disposition": f'attachment; filename="hivemind-{uid}.json"'})

    @app.post("/api/pause")
    async def api_pause():
        need_hive().pause()
        return {"ok": True}

    @app.post("/api/resume")
    async def api_resume():
        need_hive().resume()
        return {"ok": True}

    @app.post("/api/order")
    async def api_order(body: OrderIn):
        need_hive().operator_order(body.text)
        return {"ok": True}

    @app.post("/api/drone/spawn")
    async def api_spawn(body: SpawnIn):
        agent = need_hive().spawn_drone(body.specialty, body.mission, body.focus, by="Operator")
        if agent is None:
            raise HTTPException(409, "drone limit reached")
        return {"ok": True, "id": agent.id}

    @app.post("/api/agent/{agent_id}/delete")
    async def api_delete_agent(agent_id: str):
        hive = need_hive()
        agent = hive.agents.get(agent_id)
        if agent is None:
            raise HTTPException(404, "unknown or already deleted agent")
        if agent.kind == "overmind":
            hive.kill(agent, "deleted by the operator")
            hive._spawn_overmind("Previous Overmind was deleted manually by the operator.")
        else:
            hive.kill(agent, "deleted by the operator")
        return {"ok": True}

    @app.post("/api/paper/deploy")
    async def api_deploy(body: DeployIn):
        hive = need_hive()
        if hive.paper is None:
            raise HTTPException(400, "Alpaca is not configured (set ALPACA_API_KEY / ALPACA_SECRET_KEY)")
        rec = await hive.detail(body.uid)
        if rec is None:
            raise HTTPException(404, "unknown strategy")
        try:
            await hive.paper.deploy(rec["spec"])
        except Exception as exc:
            raise HTTPException(400, str(exc)) from None
        return {"ok": True}

    @app.post("/api/paper/stop")
    async def api_paper_stop():
        hive = need_hive()
        if hive.paper is None:
            raise HTTPException(400, "Alpaca is not configured")
        await hive.paper.stop(flatten=True)
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        if not authed(websocket.cookies):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        q = None
        try:
            while state["bus"] is None and state["boot"]["error"] is None:
                await websocket.send_json({"type": "snapshot", "data": snapshot()})
                await asyncio.sleep(1.0)
            bus: Bus | None = state["bus"]
            if bus is None:
                await websocket.send_json({"type": "snapshot", "data": snapshot()})
                return
            q = bus.subscribe()
            await websocket.send_json({"type": "hello", "events": list(bus.events)[-200:], "data": snapshot()})
            last = time.time()
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=0.6)
                    batch = [ev]
                    while not q.empty() and len(batch) < 100:
                        batch.append(q.get_nowait())
                    await websocket.send_json({"type": "events", "events": batch})
                except asyncio.TimeoutError:
                    pass
                if time.time() - last >= 1.2:
                    await websocket.send_json({"type": "snapshot", "data": snapshot()})
                    last = time.time()
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            if q is not None and state["bus"] is not None:
                state["bus"].unsubscribe(q)

    return app
