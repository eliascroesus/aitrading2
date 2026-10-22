"""Entry point: ``python -m hive``"""
from __future__ import annotations

import logging
import os
import threading
import webbrowser


def main() -> None:
    import uvicorn

    from .config import load_config
    from .server import create_app

    cfg = load_config()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    mode = "Claude agents" if not cfg.offline else "OFFLINE mode (no ANTHROPIC_API_KEY): procedural agents"
    print(
        "\n  HIVEMIND\n"
        f"  objective : {cfg.prop_preset}\n"
        f"  agents    : {mode}\n"
        f"  data      : {cfg.resolved_data_source} | {', '.join(cfg.instruments)} | {cfg.timeframes} min\n"
        f"  dashboard : http://{cfg.host}:{cfg.port}\n"
    )
    if cfg.host not in ("127.0.0.1", "localhost") and not cfg.dashboard_password:
        print("  WARNING   : dashboard is reachable from the network without a password; set DASHBOARD_PASSWORD\n")
    if os.getenv("HIVE_OPEN_BROWSER") == "1":
        url = f"http://127.0.0.1:{cfg.port}"
        threading.Timer(2.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level="warning", proxy_headers=True, forwarded_allow_ips="*")


if __name__ == "__main__":
    main()
