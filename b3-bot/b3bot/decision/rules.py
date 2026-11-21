"""Decisor 1 de 3: regras classicas. O BASELINE.

Este e o decisor mais importante do projeto, apesar de ser o mais simples.
Ele existe para responder a unica pergunta que valida um modelo de ML ou
um LLM: "o modelo bate uma estrategia boba, liquido de custos?".

Se o ML nao bater isso, voce nao tem modelo - tem overfitting.

Estrategia: rompimento com confirmacao de tendencia.
  - opera somente nos regimes configurados (default: tendencia no 5min);
  - exige ATR minimo (mercado parado nao paga o custo);
  - compra no rompimento da maxima das N barras anteriores, com EMA
    rapida acima da lenta, preco acima do VWAP e RSI fora da exaustao;
  - venda e o espelho.
"""
from __future__ import annotations

import math
from typing import Any

from ..features import FeatureSnapshot
from .base import Decisor, Signal, hold


def _finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


class RulesDecisor(Decisor):
    name = "rules"

    def __init__(self, params: dict[str, Any] | None = None,
                 allow_long: bool = True, allow_short: bool = False) -> None:
        p = params or {}
        self.rsi_overbought = float(p.get("rsi_overbought", 72))
        self.rsi_oversold = float(p.get("rsi_oversold", 28))
        self.require_regime = tuple(p.get("require_regime", ("trend_up", "trend_down")))
        self.min_atr_points = float(p.get("min_atr_points", 30))
        self.breakout_lookback = int(p.get("breakout_lookback", 20))
        self.allow_long = allow_long
        self.allow_short = allow_short

    def warmup_bars(self) -> int:
        return max(60, self.breakout_lookback + 30)

    def decide(self, snap: FeatureSnapshot) -> Signal:
        # --- filtros que valem para os dois lados ---------------------------
        if not _finite(snap.atr) or snap.atr <= 0:
            return hold(self.name, "ATR indisponivel (warmup)")

        if snap.atr < self.min_atr_points:
            return hold(self.name, f"ATR {snap.atr:.0f} < minimo {self.min_atr_points:.0f}: mercado parado")

        if self.require_regime and snap.regime not in self.require_regime:
            return hold(self.name, f"regime {snap.regime} fora de {list(self.require_regime)}")

        brk_up = snap.get("breakout_up_atr")
        brk_dn = snap.get("breakout_dn_atr")
        ema_spread = snap.get("ema_spread_atr")
        rsi_v = snap.get("rsi")
        vwap_dist = snap.get("dist_vwap_atr")

        for label, v in (("breakout", brk_up), ("ema_spread", ema_spread), ("rsi", rsi_v)):
            if not _finite(v):
                return hold(self.name, f"feature {label} indisponivel (warmup)")

        # --- compra ---------------------------------------------------------
        if self.allow_long and snap.regime == "trend_up":
            motivos_contra = []
            if brk_up <= 0:
                motivos_contra.append(f"sem rompimento (faltam {-brk_up:.2f} atr)")
            if ema_spread <= 0:
                motivos_contra.append("EMA rapida abaixo da lenta")
            if rsi_v >= self.rsi_overbought:
                motivos_contra.append(f"RSI {rsi_v:.0f} esticado")
            if _finite(vwap_dist) and vwap_dist < 0:
                motivos_contra.append("abaixo do VWAP")

            if not motivos_contra:
                conf = self._confidence(brk_up, ema_spread, rsi_v, vwap_dist)
                return Signal(
                    "BUY", conf,
                    f"rompimento de {self.breakout_lookback} barras (+{brk_up:.2f} atr), "
                    f"EMA+{ema_spread:.2f} atr, RSI {rsi_v:.0f}, tendencia de alta no 5min",
                    self.name,
                    {"breakout_atr": brk_up, "ema_spread_atr": ema_spread, "rsi": rsi_v},
                )
            setup_long = motivos_contra
        else:
            setup_long = ["compra desabilitada ou regime nao e de alta"]

        # --- venda ----------------------------------------------------------
        if self.allow_short and snap.regime == "trend_down" and _finite(brk_dn):
            motivos_contra = []
            if brk_dn <= 0:
                motivos_contra.append(f"sem rompimento de baixa (faltam {-brk_dn:.2f} atr)")
            if ema_spread >= 0:
                motivos_contra.append("EMA rapida acima da lenta")
            if rsi_v <= self.rsi_oversold:
                motivos_contra.append(f"RSI {rsi_v:.0f} esticado")
            if _finite(vwap_dist) and vwap_dist > 0:
                motivos_contra.append("acima do VWAP")

            if not motivos_contra:
                conf = self._confidence(brk_dn, -ema_spread, 100 - rsi_v, -vwap_dist)
                return Signal(
                    "SELL", conf,
                    f"rompimento de baixa (+{brk_dn:.2f} atr), EMA{ema_spread:.2f} atr, "
                    f"RSI {rsi_v:.0f}, tendencia de baixa no 5min",
                    self.name,
                    {"breakout_atr": brk_dn, "ema_spread_atr": ema_spread, "rsi": rsi_v},
                )

        return hold(self.name, "; ".join(setup_long[:2]))

    def _confidence(self, breakout: float, ema_spread: float, rsi_v: float,
                    vwap_dist: float) -> float:
        """Confianca heuristica em [0.5, 0.95].

        Nao e probabilidade. E uma nota de qualidade do setup, usada para
        comparar sinais entre si e para o filtro hibrido. Nao interprete
        0.8 como "80% de acerto".
        """
        score = 0.5
        score += min(0.20, max(0.0, breakout) * 0.40)     # rompimento forte
        score += min(0.15, max(0.0, ema_spread) * 0.15)   # tendencia alinhada
        if _finite(vwap_dist):
            score += min(0.10, max(0.0, vwap_dist) * 0.05)
        if 45 <= rsi_v <= 65:                              # nem esticado nem fraco
            score += 0.05
        return float(min(0.95, max(0.5, score)))


def is_rule_setup(snap: FeatureSnapshot, params: dict[str, Any] | None = None,
                  allow_long: bool = True, allow_short: bool = False) -> Signal:
    """Usado pelo hibrido e pelo LLM (only_on_rule_setup) para saber se ha
    setup antes de gastar uma inferencia."""
    return RulesDecisor(params, allow_long, allow_short).safe_decide(snap)
