"""FastAPI server: live websocket feed + REST control for the dashboard."""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
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
