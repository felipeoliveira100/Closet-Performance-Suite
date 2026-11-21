"""Carga e validacao da configuracao.

Regra de ouro deste modulo: e aqui que o robo se recusa a subir em modo
live sem confirmacao explicita. Nao remova essa checagem.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from pathlib import Path
from typing import Any, Literal

import yaml

Mode = Literal["paper", "demo", "live"]
LIVE_PHRASE = "EU ASSUMO O RISCO"


class ConfigError(Exception):
    """Configuracao invalida. O robo nao sobe."""


def _parse_hhmm(value: str, label: str) -> time:
    try:
        hh, mm = str(value).split(":")
        return time(int(hh), int(mm))
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"{label}: horario invalido {value!r}, use HH:MM") from exc


@dataclass(frozen=True)
class SymbolConfig:
    root: str = "WIN"
    override: str | None = None
    point_value: float = 0.20
    tick_size: float = 5.0
    min_volume: int = 1

    def __post_init__(self) -> None:
        if self.point_value <= 0:
            raise ConfigError("symbol.point_value deve ser > 0")
        if self.tick_size <= 0:
            raise ConfigError("symbol.tick_size deve ser > 0")


@dataclass(frozen=True)
class MT5Config:
    login: int | None = None
    password: str | None = None
    server: str | None = None
    terminal_path: str | None = None
    magic: int = 20261121
    deviation_points: int = 20


@dataclass(frozen=True)
class SessionConfig:
    timezone: str = "America/Sao_Paulo"
    entry_start: time = time(9, 15)
    entry_end: time = time(17, 15)
    force_flat_at: time = time(17, 40)

    def __post_init__(self) -> None:
        if not (self.entry_start < self.entry_end < self.force_flat_at):
            raise ConfigError(
                "session: exige entry_start < entry_end < force_flat_at "
                f"(recebi {self.entry_start} / {self.entry_end} / {self.force_flat_at})"
            )


@dataclass(frozen=True)
class RiskConfig:
    max_contracts: int = 1
    risk_per_trade_brl: float = 150.0
    stop_atr_mult: float = 1.5
    target_atr_mult: float = 2.5
    min_stop_points: int = 100
    max_stop_points: int = 600
    max_daily_loss_brl: float = 400.0
    max_daily_trades: int = 8
    max_consecutive_losses: int = 3
    cooldown_bars_after_exit: int = 3
    min_rr_after_costs: float = 1.2

    def __post_init__(self) -> None:
        if self.max_contracts < 1:
            raise ConfigError("risk.max_contracts deve ser >= 1")
        if self.max_daily_loss_brl <= 0:
            raise ConfigError(
                "risk.max_daily_loss_brl deve ser > 0 - o kill switch nao e opcional"
            )
        if self.min_stop_points >= self.max_stop_points:
            raise ConfigError("risk.min_stop_points deve ser < max_stop_points")
        if self.target_atr_mult <= self.stop_atr_mult:
            raise ConfigError(
                "risk.target_atr_mult <= stop_atr_mult: payoff negativo por desenho. "
                "Se isso e intencional, voce precisa de acerto muito alto - revise."
            )


@dataclass(frozen=True)
class CostsConfig:
    brokerage_per_order_brl: float = 0.0
    exchange_fee_per_contract_brl: float = 0.27
    registration_fee_per_order_brl: float = 0.10
    iss_rate: float = 0.0
    daytrade_ir_rate: float = 0.20
    daytrade_irrf_rate: float = 0.01


@dataclass(frozen=True)
class DirectionConfig:
    allow_long: bool = True
    allow_short: bool = False

    def __post_init__(self) -> None:
        if not self.allow_long and not self.allow_short:
            raise ConfigError("direction: pelo menos um lado precisa estar habilitado")


@dataclass(frozen=True)
class LoggingConfig:
    journal_path: str = "logs/journal.jsonl"
    level: str = "INFO"


@dataclass(frozen=True)
class Config:
    mode: Mode
    symbol: SymbolConfig
    mt5: MT5Config
    session: SessionConfig
    risk: RiskConfig
    costs: CostsConfig
    direction: DirectionConfig
    logging: LoggingConfig
    decisor: dict[str, Any] = field(default_factory=dict)
    live_confirmation: str = ""

    @property
    def active_decisor(self) -> str:
        return str(self.decisor.get("active", "rules"))

    @property
    def sends_real_orders(self) -> bool:
        """True quando ordens saem de fato para a corretora (demo ou real)."""
        return self.mode in ("demo", "live")

    def decisor_params(self, name: str | None = None) -> dict[str, Any]:
        key = name or self.active_decisor
        return dict(self.decisor.get(key, {}) or {})


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config nao encontrado: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    mode = str(raw.get("mode", "paper")).lower()
    if mode not in ("paper", "demo", "live"):
        raise ConfigError(f"mode invalido: {mode!r}. Use paper, demo ou live.")

    confirmation = str(raw.get("live_confirmation", "") or "").strip()
    if mode == "live" and confirmation != LIVE_PHRASE:
        raise ConfigError(
            "Modo live bloqueado.\n"
            "  Para operar DINHEIRO REAL, escreva no config.yaml:\n"
            f'    live_confirmation: "{LIVE_PHRASE}"\n'
            "  Antes disso, rode em paper e depois em demo e audite o journal."
        )

    sess_raw = raw.get("session", {}) or {}
    session = SessionConfig(
        timezone=sess_raw.get("timezone", "America/Sao_Paulo"),
        entry_start=_parse_hhmm(sess_raw.get("entry_start", "09:15"), "session.entry_start"),
        entry_end=_parse_hhmm(sess_raw.get("entry_end", "17:15"), "session.entry_end"),
        force_flat_at=_parse_hhmm(sess_raw.get("force_flat_at", "17:40"), "session.force_flat_at"),
    )

    cfg = Config(
        mode=mode,  # type: ignore[arg-type]
        live_confirmation=confirmation,
        symbol=SymbolConfig(**(raw.get("symbol", {}) or {})),
        mt5=MT5Config(**(raw.get("mt5", {}) or {})),
        session=session,
        risk=RiskConfig(**(raw.get("risk", {}) or {})),
        costs=CostsConfig(**(raw.get("costs", {}) or {})),
        direction=DirectionConfig(**(raw.get("direction", {}) or {})),
        logging=LoggingConfig(**(raw.get("logging", {}) or {})),
        decisor=raw.get("decisor", {}) or {},
    )

    valid_decisors = {"rules", "ml", "llm", "hybrid"}
    if cfg.active_decisor not in valid_decisors:
        raise ConfigError(
            f"decisor.active invalido: {cfg.active_decisor!r}. Use um de {sorted(valid_decisors)}"
        )
    return cfg
