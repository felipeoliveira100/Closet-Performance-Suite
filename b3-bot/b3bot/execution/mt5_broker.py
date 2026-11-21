"""MT5Broker: envia ordem de verdade pelo terminal MetaTrader 5. WINDOWS.

Pegadinhas do order_send tratadas aqui - cada uma ja fez robo de alguem
"operar" sem posicao nenhuma:

1) order_send PODE DEVOLVER None. Nao e excecao, nao e retcode de erro: e
   None. Ler .retcode de None levanta AttributeError no meio do pregao.
   Checamos None primeiro e consultamos last_error().

2) FILLING MODE E MASCARA DE BITS, E OS ENUMS NAO SAO INTERCAMBIAVEIS.
   symbol_info.filling_mode e uma mascara (SYMBOL_FILLING_FOK=1,
   SYMBOL_FILLING_IOC=2). O campo type_filling da ordem usa OUTRO enum
   (ORDER_FILLING_FOK=0, ORDER_FILLING_IOC=1, ...). Passar a mascara
   direto no type_filling e o caminho mais curto para o retcode 10030
   ("modo de preenchimento invalido"). Resolvemos pela mascara e
   traduzimos para o enum certo.

3) ORDEM ACEITA != ORDEM EXECUTADA. retcode DONE com volume parcial
   acontece. Conferimos o volume efetivamente preenchido.

4) RECONCILIACAO. A posicao real vem sempre de positions_get, filtrada
   pelo magic do robo, nunca da memoria do processo.

5) AUTOTRADING DESLIGADO. O botao "Algo Trading" do terminal desligado faz
   toda ordem ser recusada. Checamos antes de tentar.
"""
from __future__ import annotations

import logging
from typing import Any

from ..config import Config
from .base import Broker, BrokerPosition, OrderResult, Side

logger = logging.getLogger(__name__)

try:  # pragma: no cover - so existe no Windows
    import MetaTrader5 as mt5  # type: ignore
except ImportError:  # pragma: no cover
    mt5 = None  # type: ignore


class MT5BrokerError(RuntimeError):
    pass


def resolve_filling_mode(symbol_info: Any, mt5_module: Any) -> int:
    """Traduz a MASCARA do simbolo para o ENUM de type_filling.

    Dois espacos de valores diferentes, e eles NAO coincidem:
        mascara do simbolo: SYMBOL_FILLING_FOK=1, SYMBOL_FILLING_IOC=2
        enum da ordem:      ORDER_FILLING_FOK=0, ORDER_FILLING_IOC=1, ...
    """
    mask = int(getattr(symbol_info, "filling_mode", 0) or 0)
    fok_bit = int(getattr(mt5_module, "SYMBOL_FILLING_FOK", 1))
    ioc_bit = int(getattr(mt5_module, "SYMBOL_FILLING_IOC", 2))

    if mask & ioc_bit:
        # IOC primeiro: aceita preenchimento parcial, o que em futuro da B3
        # e preferivel a ter a ordem inteira recusada por falta de contraparte.
        return int(mt5_module.ORDER_FILLING_IOC)
    if mask & fok_bit:
        return int(mt5_module.ORDER_FILLING_FOK)
    return int(getattr(mt5_module, "ORDER_FILLING_RETURN", 2))


class MT5Broker(Broker):
    name = "mt5"
    sends_real_orders = True

    def __init__(self, cfg: Config, symbol: str) -> None:
        if mt5 is None:
            raise MT5BrokerError(
                "pacote MetaTrader5 ausente. Execucao real exige Windows + terminal MT5."
            )
        self.cfg = cfg
        self.symbol = symbol
        self.magic = int(cfg.mt5.magic)
        self.deviation = int(cfg.mt5.deviation_points)
        self._filling: int | None = None
        self._is_demo: bool | None = None

    # --- conexao ------------------------------------------------------------

    def connect(self) -> None:
        info = mt5.terminal_info()
        acct = mt5.account_info()
        if info is None or acct is None:
            raise MT5BrokerError(
                f"terminal MT5 nao inicializado: {mt5.last_error()}. "
                "Conecte o feed (MT5Feed.connect) antes do broker."
            )
        if not info.trade_allowed:
            raise MT5BrokerError(
                "AutoTrading desligado no terminal. Ligue o botao 'Algo Trading' no MT5 - "
                "sem isso toda ordem e recusada."
            )
        sinfo = mt5.symbol_info(self.symbol)
        if sinfo is None:
            raise MT5BrokerError(f"symbol_info({self.symbol}) vazio: {mt5.last_error()}")
        if getattr(sinfo, "trade_mode", None) == getattr(mt5, "SYMBOL_TRADE_MODE_DISABLED", -1):
            raise MT5BrokerError(f"{self.symbol}: negociacao desabilitada neste simbolo")

        self._filling = resolve_filling_mode(sinfo, mt5)
        self._is_demo = getattr(acct, "trade_mode", None) == 0

        # A trava final: config em live + conta real exige a frase de confirmacao,
        # que load_config ja validou. Aqui so avisamos de forma inequivoca.
        tipo = "DEMO" if self._is_demo else "*** CONTA REAL ***"
        if self.cfg.mode == "demo" and not self._is_demo:
            raise MT5BrokerError(
                "config diz mode: demo, mas o terminal esta logado numa CONTA REAL. "
                "Troque para a conta demo no MT5 ou mude o mode para live (com a frase de confirmacao)."
            )
        logger.warning(
            "MT5Broker ARMADO | %s | conta %s %s | saldo %.2f | magic %d | filling %s",
            self.symbol, acct.login, tipo, acct.balance, self.magic, self._filling,
        )

    def shutdown(self) -> None:
        pos = self.current_position()
        if pos is not None:
            logger.critical(
                "MT5Broker encerrado COM POSICAO ABERTA: %s %dx @ %.0f. "
                "O robo nao esta mais cuidando dela - zere manualmente se for o caso.",
                pos.side, pos.contracts, pos.entry_price,
            )

    # --- ordens -------------------------------------------------------------

    def _send(self, request: dict[str, Any], what: str) -> OrderResult:
        logger.info("order_send (%s): %s", what, request)
        result = mt5.order_send(request)

        # (1) None: nao e excecao e nao tem retcode.
        if result is None:
            err = mt5.last_error()
            logger.error("order_send devolveu None (%s): %s", what, err)
            return OrderResult(False, retcode=None,
                               comment=f"order_send None: {err}", raw={"last_error": str(err)})

        retcode = int(getattr(result, "retcode", -1))
        done = int(getattr(mt5, "TRADE_RETCODE_DONE", 10009))
        partial = int(getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010))
        raw = {k: getattr(result, k, None) for k in
               ("retcode", "deal", "order", "volume", "price", "bid", "ask", "comment",
                "request_id", "retcode_external")}

        if retcode not in (done, partial):
            logger.error("ordem recusada (%s): retcode=%s %s",
                         what, retcode, getattr(result, "comment", ""))
            return OrderResult(False, retcode=retcode,
                               comment=str(getattr(result, "comment", "")), raw=raw)

        filled = float(getattr(result, "volume", 0.0) or 0.0)
        requested = float(request.get("volume", 0.0))
        if retcode == partial or (requested and filled < requested):
            # (3) aceita != executada por completo
            logger.warning("preenchimento PARCIAL (%s): %.0f de %.0f contratos",
                           what, filled, requested)

        return OrderResult(
            ok=True,
            side=request.get("_side"),
            contracts=int(filled) if filled else int(requested),
            filled_price=float(getattr(result, "price", 0.0) or 0.0),
            retcode=retcode,
            comment=str(getattr(result, "comment", "")),
            order_id=int(getattr(result, "order", 0) or 0),
            raw=raw,
        )

    def market_order(self, side: Side, contracts: int, price_hint: float,
                     comment: str = "") -> OrderResult:
        if contracts <= 0:
            return OrderResult(False, comment="contratos <= 0")
        if (existing := self.current_position()) is not None:
            return OrderResult(False, comment=f"posicao ja aberta no broker: {existing.side}")

        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return OrderResult(False, comment=f"sem tick para {self.symbol}: {mt5.last_error()}")
        price = float(tick.ask if side == "BUY" else tick.bid)

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(contracts),
            "type": mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": (comment or "b3bot")[:31],  # MT5 trunca em 31 chars
            "type_time": mt5.ORDER_TIME_DAY,       # day trade: nao sobrevive ao dia
            "type_filling": self._filling,
            "_side": side,
        }
        return self._send(req, f"abrir {side}")

    def close_all(self, price_hint: float, comment: str = "") -> OrderResult:
        pos = self.current_position()
        if pos is None:
            return OrderResult(False, comment="sem posicao para zerar")

        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return OrderResult(False, comment=f"sem tick para zerar: {mt5.last_error()}")

        close_side: Side = "SELL" if pos.side == "BUY" else "BUY"
        price = float(tick.bid if close_side == "SELL" else tick.ask)
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(pos.contracts),
            "type": mt5.ORDER_TYPE_SELL if close_side == "SELL" else mt5.ORDER_TYPE_BUY,
            "position": pos.ticket,   # fecha ESTA posicao, nao abre oposta
            "price": price,
            "deviation": max(self.deviation, 40),  # zeragem: aceita mais slippage
            "magic": self.magic,
            "comment": (comment or "b3bot close")[:31],
            "type_time": mt5.ORDER_TIME_DAY,
            "type_filling": self._filling,
            "_side": close_side,
        }
        res = self._send(req, "zerar")
        if not res.ok:
            logger.critical(
                "FALHA AO ZERAR posicao %s %dx. Isso deixa risco aberto - "
                "zere manualmente no terminal se persistir.", pos.side, pos.contracts,
            )
        return res

    # --- reconciliacao ------------------------------------------------------

    def current_position(self) -> BrokerPosition | None:
        """(4) fonte da verdade: o broker, filtrado pelo magic do robo."""
        positions = mt5.positions_get(symbol=self.symbol)
        if not positions:
            return None
        mine = [p for p in positions if int(getattr(p, "magic", -1)) == self.magic]
        if not mine:
            others = len(positions)
            logger.warning(
                "%d posicao(oes) em %s com outro magic - nao sao deste robo, ignorando",
                others, self.symbol,
            )
            return None
        if len(mine) > 1:
            logger.error("%d posicoes do robo no mesmo simbolo - estado inesperado", len(mine))
        p = mine[0]
        side: Side = "BUY" if int(p.type) == int(mt5.POSITION_TYPE_BUY) else "SELL"
        return BrokerPosition(
            side=side,
            contracts=int(p.volume),
            entry_price=float(p.price_open),
            unrealized_brl=float(getattr(p, "profit", 0.0) or 0.0),
            ticket=int(p.ticket),
        )
