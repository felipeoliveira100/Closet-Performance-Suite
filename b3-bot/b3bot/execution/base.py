"""Interface de execucao. Duas implementacoes: PaperBroker e MT5Broker.

O engine conversa somente com esta interface, e por isso o mesmo codigo de
decisao e de risco roda em paper, demo e real sem alteracao. O que muda e
qual broker foi injetado - o que significa que o caminho testado em paper
e literalmente o mesmo que executa com dinheiro.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

Side = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class OrderResult:
    ok: bool
    side: Side | None = None
    contracts: int = 0
    filled_price: float | None = None
    retcode: int | None = None
    comment: str = ""
    order_id: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.ok:
            return (f"OK {self.side} {self.contracts}x @ {self.filled_price:.0f}"
                    f" (ordem {self.order_id})")
        return f"FALHOU retcode={self.retcode} {self.comment}"


@dataclass(frozen=True)
class BrokerPosition:
    """Posicao como o BROKER a reporta - fonte da verdade.

    O engine mantem seu proprio objeto Position, mas reconcilia contra
    isto. Divergencia entre os dois significa que algo foi executado fora
    do robo (ordem manual, stop da corretora, liquidacao) e o robo precisa
    parar em vez de operar sobre uma crenca errada.
    """

    side: Side
    contracts: int
    entry_price: float
    unrealized_brl: float = 0.0
    ticket: int | None = None


class Broker(ABC):
    name: str = "base"
    sends_real_orders: bool = False

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def shutdown(self) -> None: ...

    @abstractmethod
    def market_order(self, side: Side, contracts: int, price_hint: float,
                     comment: str = "") -> OrderResult:
        """Ordem a mercado para ABRIR posicao."""

    @abstractmethod
    def close_all(self, price_hint: float, comment: str = "") -> OrderResult:
        """Zera a posicao a mercado."""

    @abstractmethod
    def current_position(self) -> BrokerPosition | None:
        """Posicao que o broker reporta agora."""

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()
