"""Decisor 4: hibrido - regras propõem, ML veta.

Na minha opiniao e o desenho mais defensavel dos quatro para operar de
verdade, e por um motivo pratico: quando ele erra, voce consegue descobrir
POR QUE. A entrada tem uma causa legivel (o rompimento) e a aprovacao tem
um numero (a probabilidade do modelo). Um modelo end-to-end que decide
tudo sozinho erra sem explicacao, e voce fica sem saber se conserta a
feature, o rotulo ou a estrategia.

Fluxo: regras acham o setup -> ML estima P(movimento favoravel) ->
abaixo do limiar, veta. O ML aqui nunca ABRE posicao por conta propria;
so aprova ou derruba o que as regras propuseram.
"""
from __future__ import annotations

import math
from typing import Any

from ..features import FeatureSnapshot
from .base import Decisor, Signal, hold
from .ml_local import MLDecisor
from .rules import RulesDecisor


class HybridDecisor(Decisor):
    name = "hybrid"

    def __init__(self, params: dict[str, Any] | None = None,
                 rules_params: dict[str, Any] | None = None,
                 ml_params: dict[str, Any] | None = None,
                 allow_long: bool = True, allow_short: bool = False) -> None:
        p = params or {}
        self.veto_threshold = float(p.get("ml_veto_threshold", 0.45))
        self.require_ml = bool(p.get("require_ml", True))
        self.rules = RulesDecisor(rules_params, allow_long, allow_short)
        self.ml = MLDecisor(ml_params, allow_long, allow_short)

    def warmup_bars(self) -> int:
        return max(self.rules.warmup_bars(), self.ml.warmup_bars())

    def decide(self, snap: FeatureSnapshot) -> Signal:
        base = self.rules.safe_decide(snap)
        if not base.is_entry:
            return hold(self.name, f"regras: {base.reason}")

        if not self.ml.ready:
            if self.require_ml:
                # Sem modelo e com veto obrigatorio: nao opera. Deliberado -
                # "sem filtro" nao deve virar "opera sem filtro" em silencio.
                return hold(self.name, "modelo de ML indisponivel e veto obrigatorio (require_ml)")
            return Signal(base.action, base.confidence,
                          f"{base.reason} | SEM filtro de ML (modelo ausente)",
                          self.name, {**base.meta, "ml_available": False})

        p_up = self.ml.score_up(snap)
        if not math.isfinite(p_up):
            return hold(self.name, "ML nao pontuou este candle (features faltando)")

        # Para venda, a probabilidade favoravel e a de NAO subir.
        p_favor = p_up if base.action == "BUY" else 1.0 - p_up
        meta = {**base.meta, "p_up": round(p_up, 4), "p_favor": round(p_favor, 4),
                "veto_threshold": self.veto_threshold}

        if p_favor < self.veto_threshold:
            return hold(self.name,
                        f"ML VETOU {base.action}: P(favoravel)={p_favor:.2f} "
                        f"< {self.veto_threshold:.2f} | setup era: {base.reason}",
                        **meta)

        # Confianca final: media geometrica - exige que AMBOS concordem.
        # Media aritmetica deixaria um sinal fraco passar carregado pelo outro.
        conf = float(min(0.95, max(0.0, (base.confidence * p_favor) ** 0.5)))
        return Signal(base.action, conf,
                      f"{base.reason} | ML aprovou: P(favoravel)={p_favor:.2f}",
                      self.name, meta)
