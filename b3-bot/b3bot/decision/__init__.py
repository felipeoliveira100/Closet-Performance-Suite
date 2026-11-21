"""Decisores: regras, ML local, LLM e hibrido - todos sob a mesma interface."""
from __future__ import annotations

from ..config import Config
from .base import Action, Decisor, Signal, hold
from .hybrid import HybridDecisor
from .llm import LLMDecisor
from .ml_local import MLDecisor
from .rules import RulesDecisor

__all__ = [
    "Action", "Decisor", "Signal", "hold",
    "RulesDecisor", "MLDecisor", "LLMDecisor", "HybridDecisor",
    "build_decisor", "build_all_decisors",
]


def build_decisor(cfg: Config, name: str | None = None) -> Decisor:
    """Constroi o decisor pedido a partir da config."""
    name = (name or cfg.active_decisor).lower()
    long_, short_ = cfg.direction.allow_long, cfg.direction.allow_short

    if name == "rules":
        return RulesDecisor(cfg.decisor_params("rules"), long_, short_)
    if name == "ml":
        return MLDecisor(cfg.decisor_params("ml"), long_, short_)
    if name == "llm":
        return LLMDecisor(cfg.decisor_params("llm"), long_, short_,
                          rules_params=cfg.decisor_params("rules"))
    if name == "hybrid":
        return HybridDecisor(cfg.decisor_params("hybrid"), cfg.decisor_params("rules"),
                             cfg.decisor_params("ml"), long_, short_)
    raise ValueError(f"decisor desconhecido: {name!r}")


def build_all_decisors(cfg: Config, include_llm: bool = False) -> dict[str, Decisor]:
    """Todos os decisores para rodar em paralelo e comparar.

    include_llm=False por padrao: no backtest o LLM custaria uma chamada de
    API por candle e, por nao ser deterministico, o resultado nao seria
    reproduzivel. Compare-o em paper trading ao vivo.
    """
    names = ["rules", "ml", "hybrid"] + (["llm"] if include_llm else [])
    return {n: build_decisor(cfg, n) for n in names}
