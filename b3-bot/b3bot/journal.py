"""Journal de auditoria: toda decisao do robo vira uma linha JSON.

Isso nao e log de debug. E o registro que te permite responder, depois
de um dia ruim, a pergunta que importa: o robo decidiu errado ou
executou errado? Sem isso voce esta apostando, nao operando.

Formato JSONL (uma linha por evento) para abrir direto no pandas:
    df = pd.read_json("logs/journal.jsonl", lines=True)
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "item"):  # numpy scalar
        try:
            return obj.item()
        except Exception:  # noqa: BLE001
            return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


class Journal:
    """Append-only, thread-safe, com flush imediato.

    Flush imediato e deliberado: se o processo morrer no meio do pregao,
    as decisoes anteriores precisam estar no disco.
    """

    def __init__(self, path: str | Path, mode: str = "paper", run_id: str | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._lock = threading.Lock()
        self.write("run_start", {"mode": mode, "run_id": self.run_id})

    def write(self, event: str, payload: dict[str, Any] | None = None) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "mode": self.mode,
            "event": event,
            **_jsonable(payload or {}),
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()

    # --- atalhos semanticos -------------------------------------------------

    def decision(self, signal: Any, context: dict[str, Any] | None = None) -> None:
        self.write("decision", {"signal": signal, "context": context or {}})

    def order_attempt(self, request: dict[str, Any]) -> None:
        self.write("order_attempt", {"request": request})

    def order_result(self, ok: bool, detail: dict[str, Any]) -> None:
        self.write("order_result", {"ok": ok, "detail": detail})

    def risk_block(self, reason: str, detail: dict[str, Any] | None = None) -> None:
        self.write("risk_block", {"reason": reason, **(detail or {})})

    def position_closed(self, detail: dict[str, Any]) -> None:
        self.write("position_closed", detail)

    def kill_switch(self, reason: str, detail: dict[str, Any] | None = None) -> None:
        logger.critical("KILL SWITCH: %s", reason)
        self.write("kill_switch", {"reason": reason, **(detail or {})})

    def error(self, where: str, exc: BaseException) -> None:
        logger.exception("erro em %s", where)
        self.write("error", {"where": where, "type": type(exc).__name__, "message": str(exc)})


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
