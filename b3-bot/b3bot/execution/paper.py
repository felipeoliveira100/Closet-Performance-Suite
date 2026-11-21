"""PaperBroker: simula execucao sem mandar nada para a corretora.

Nao e um mock vazio - ele modela as duas coisas que mais separam backtest
de realidade:

  SLIPPAGE: ordem a mercado nao executa no preco que voce viu. Compra
  paga o lado do ask (ou pior), venda recebe o bid. O default assume meio
  spread + 1 tick contra voce, o que em WIN e otimista num rompimento e
  pessimista num mercado calmo.

  CUSTO: cobrado nas duas pontas, pelo mesmo CostModel do backtest e do
  relatorio - nunca por uma conta separada que divergiria.
"""
from __future__ import annotations

import logging
from itertools import count
from typing import Any

from ..config import Config
from ..costs import CostModel
from .base import Broker, BrokerPosition, OrderResult, Side

logger = logging.getLogger(__name__)


class PaperBroker(Broker):
    name = "paper"
    sends_real_orders = False

    def __init__(self, cfg: Config, slippage_ticks: float = 1.0) -> None:
        self.cfg = cfg
        self.costs = CostModel(cfg.costs, cfg.symbol)
        self.tick = cfg.symbol.tick_size
        self.point_value = cfg.symbol.point_value
        self.slippage_ticks = slippage_ticks
        self._pos: BrokerPosition | None = None
        self._ids = count(1)
        self.realized_brl = 0.0
        self.orders: list[dict[str, Any]] = []

    def connect(self) -> None:
        logger.info("PaperBroker ativo - NENHUMA ordem sai para a corretora")

    def shutdown(self) -> None:
        if self._pos is not None:
            logger.warning("PaperBroker encerrado com posicao simulada aberta: %s", self._pos)

    def _fill_price(self, side: Side, price_hint: float) -> float:
        """Preco de execucao com slippage contra o robo."""
        adverse = self.slippage_ticks * self.tick
        raw = price_hint + adverse if side == "BUY" else price_hint - adverse
        return round(raw / self.tick) * self.tick

    def market_order(self, side: Side, contracts: int, price_hint: float,
                     comment: str = "") -> OrderResult:
        if contracts <= 0:
            return OrderResult(False, comment="contratos <= 0")
        if self._pos is not None:
            return OrderResult(False, comment=f"ja existe posicao aberta: {self._pos.side}")

        fill = self._fill_price(side, price_hint)
        oid = next(self._ids)
        self._pos = BrokerPosition(side=side, contracts=contracts, entry_price=fill, ticket=oid)
        rec = {"id": oid, "action": "open", "side": side, "contracts": contracts,
               "price_hint": price_hint, "fill": fill, "comment": comment}
        self.orders.append(rec)
        logger.info("[PAPER] abriu %s %dx @ %.0f (hint %.0f, slippage %.0f pts)",
                    side, contracts, fill, price_hint, abs(fill - price_hint))
        return OrderResult(True, side, contracts, fill, retcode=0,
                           comment="paper fill", order_id=oid, raw=rec)

    def close_all(self, price_hint: float, comment: str = "") -> OrderResult:
        if self._pos is None:
            return OrderResult(False, comment="sem posicao para zerar")
        pos = self._pos
        close_side: Side = "SELL" if pos.side == "BUY" else "BUY"
        fill = self._fill_price(close_side, price_hint)
        pnl = self.costs.net_pnl_brl(pos.entry_price, fill, pos.contracts, pos.side)
        self.realized_brl += pnl
        oid = next(self._ids)
        rec = {"id": oid, "action": "close", "side": close_side, "contracts": pos.contracts,
               "price_hint": price_hint, "fill": fill, "pnl_brl": round(pnl, 2),
               "comment": comment}
        self.orders.append(rec)
        self._pos = None
        logger.info("[PAPER] zerou @ %.0f | PnL liquido R$ %+.2f | acumulado R$ %+.2f",
                    fill, pnl, self.realized_brl)
        return OrderResult(True, close_side, rec["contracts"], fill, retcode=0,
                           comment="paper close", order_id=oid, raw=rec)

    def current_position(self) -> BrokerPosition | None:
        return self._pos

    def mark_to_market(self, price: float) -> None:
        if self._pos is None:
            return
        direction = 1.0 if self._pos.side == "BUY" else -1.0
        unreal = (price - self._pos.entry_price) * direction * self._pos.contracts * self.point_value
        self._pos = BrokerPosition(self._pos.side, self._pos.contracts,
                                   self._pos.entry_price, unreal, self._pos.ticket)
