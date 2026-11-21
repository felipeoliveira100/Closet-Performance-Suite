"""Interface comum dos decisores.

Um decisor faz UMA coisa: olha o estado do mercado e opina sobre direcao.
Ele NAO decide tamanho, NAO decide stop e NAO sabe se pode operar agora -
isso e do modulo de risco. Essa separacao e o que permite trocar o
decisor (regras, ML, LLM) sem tocar na gestao de risco, e comparar os
tres sob exatamente as mesmas regras de risco e custo.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from ..features import FeatureSnapshot

Action = Literal["BUY", "SELL", "HOLD"]


@dataclass(frozen=True)
class Signal:
    """Opiniao do decisor sobre o candle fechado atual."""

    action: Action
    confidence: float          # 0.0 a 1.0
    reason: str                # legivel por humano - vai para o journal
    decisor: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action not in ("BUY", "SELL", "HOLD"):
            raise ValueError(f"action invalida: {self.action!r}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence fora de [0,1]: {self.confidence}")

    @property
    def is_entry(self) -> bool:
        return self.action in ("BUY", "SELL")

    def __str__(self) -> str:
        return f"{self.action}({self.confidence:.2f}) [{self.decisor}] {self.reason}"


def hold(decisor: str, reason: str, **meta: Any) -> Signal:
    """Atalho para o caso mais comum e mais seguro: nao fazer nada.

    Todo decisor deve devolver HOLD diante de qualquer duvida, dado
    faltante ou erro. Nunca uma entrada por omissao.
    """
    return Signal("HOLD", 0.0, reason, decisor, meta)


class Decisor(ABC):
    """Contrato: recebe snapshot, devolve Signal. Nunca levanta excecao."""

    name: str = "base"

    @abstractmethod
    def decide(self, snap: FeatureSnapshot) -> Signal:
        """Opiniao sobre o candle fechado atual.

        Implementacoes NAO devem propagar excecao: um erro no decisor tem
        de virar HOLD, nunca derrubar o robo no meio do pregao com
        posicao aberta.
        """

    def safe_decide(self, snap: FeatureSnapshot) -> Signal:
        """Envelope que garante o contrato acima."""
        try:
            sig = self.decide(snap)
            if not isinstance(sig, Signal):
                return hold(self.name, f"decisor devolveu {type(sig).__name__}, nao Signal")
            return sig
        except Exception as exc:  # noqa: BLE001
            return hold(self.name, f"erro no decisor: {type(exc).__name__}: {exc}")

    def warmup_bars(self) -> int:
        """Minimo de candles de 1min necessarios antes de confiar no decisor."""
        return 60
