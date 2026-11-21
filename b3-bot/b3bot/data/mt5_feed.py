"""Feed de market data via terminal MetaTrader 5. SOMENTE WINDOWS.

Pegadinhas tratadas aqui, todas ja custaram dinheiro a alguem:

1) BARRA EM FORMACAO. copy_rates_from_pos(..., start_pos=0, ...) inclui a
   barra atual, que ainda nao fechou. Usamos start_pos=1 para pular ela,
   e ainda aplicamos drop_forming_bar como segunda trava.

2) TIMEZONE DO SERVIDOR. O MT5 devolve horario do servidor da corretora,
   que frequentemente NAO e horario de Brasilia. Comparar com
   datetime.now() local produz decisao errada sobre "fim de pregao".
   Medimos o offset real contra o ultimo tick e expomos server_now().

3) SIMBOLO NAO SELECIONADO. Se o contrato nao esta no Market Watch,
   copy_rates devolve vazio sem erro obvio. Chamamos symbol_select.

4) CONTRATO VENCIDO. O codigo e resolvido por vencimento e conferido
   contra os simbolos que o terminal realmente oferece.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd

from ..contracts import active_contract, days_to_expiry, validate_against_mt5
from .base import Bars, MarketFeed, MarketSnapshot, drop_forming_bar

logger = logging.getLogger(__name__)

try:  # pragma: no cover - so existe no Windows
    import MetaTrader5 as mt5  # type: ignore
except ImportError:  # pragma: no cover
    mt5 = None  # type: ignore


class MT5Unavailable(RuntimeError):
    """O pacote MetaTrader5 nao esta disponivel (voce nao esta no Windows)."""


class MT5Error(RuntimeError):
    """O terminal recusou a operacao."""


def require_mt5() -> Any:
    if mt5 is None:
        raise MT5Unavailable(
            "pacote MetaTrader5 nao encontrado.\n"
            "  Execucao real exige Windows + terminal MT5 instalado e logado.\n"
            "  Em Linux/macOS use SyntheticFeed ou o backtest."
        )
    return mt5


class MT5Feed(MarketFeed):
    def __init__(self, symbol_cfg, mt5_cfg, roll_days_before: int = 2) -> None:
        self.symbol_cfg = symbol_cfg
        self.mt5_cfg = mt5_cfg
        self.roll_days_before = roll_days_before
        self.symbol: str = ""
        self._server_offset: pd.Timedelta = pd.Timedelta(0)
        self._connected = False

    # --- conexao ------------------------------------------------------------

    def connect(self) -> None:
        m = require_mt5()
        kwargs: dict[str, Any] = {}
        if self.mt5_cfg.terminal_path:
            kwargs["path"] = self.mt5_cfg.terminal_path
        if self.mt5_cfg.login:
            kwargs.update(
                login=int(self.mt5_cfg.login),
                password=self.mt5_cfg.password,
                server=self.mt5_cfg.server,
            )
        if not m.initialize(**kwargs):
            raise MT5Error(f"mt5.initialize falhou: {m.last_error()}")

        info = m.terminal_info()
        acct = m.account_info()
        if info is None or acct is None:
            raise MT5Error(f"terminal/conta indisponivel: {m.last_error()}")
        if not info.trade_allowed:
            logger.warning(
                "terminal com AutoTrading DESLIGADO - nenhuma ordem sera aceita. "
                "Ligue o botao 'Algo Trading' no MT5."
            )

        self.symbol = self._resolve_symbol()
        if not m.symbol_select(self.symbol, True):
            raise MT5Error(f"symbol_select({self.symbol}) falhou: {m.last_error()}")

        sinfo = m.symbol_info(self.symbol)
        if sinfo is None:
            raise MT5Error(f"symbol_info({self.symbol}) vazio: {m.last_error()}")

        self._measure_server_offset()
        self._connected = True
        logger.info(
            "MT5 conectado | conta %s (%s) | %s | vence em %d dias | offset servidor %s",
            acct.login,
            "DEMO" if getattr(acct, "trade_mode", 0) == 0 else "REAL",
            self.symbol,
            self._safe_dte(),
            self._server_offset,
        )

    def _safe_dte(self) -> int:
        try:
            return days_to_expiry(self.symbol)
        except ValueError:
            return -1

    def _resolve_symbol(self) -> str:
        if self.symbol_cfg.override:
            return str(self.symbol_cfg.override)
        code = active_contract(
            self.symbol_cfg.root, date.today(), roll_days_before=self.roll_days_before
        )
        return validate_against_mt5(code, self.symbol_cfg.root, mt5)

    def _measure_server_offset(self) -> None:
        """offset = hora do servidor - hora UTC. Usa o ultimo tick como relogio."""
        m = require_mt5()
        tick = m.symbol_info_tick(self.symbol)
        if tick is None or not getattr(tick, "time", 0):
            logger.warning("sem tick para medir offset do servidor; assumindo UTC")
            return
        server_time = pd.Timestamp(tick.time, unit="s")
        utc_now = pd.Timestamp.utcnow().tz_localize(None)
        raw = server_time - utc_now
        # arredonda para a hora cheia: offsets de servidor sao horas inteiras
        self._server_offset = pd.Timedelta(hours=round(raw.total_seconds() / 3600))

    def server_now(self) -> pd.Timestamp:
        """Hora atual NO FUSO DO SERVIDOR - e nela que os candles sao datados."""
        return pd.Timestamp.utcnow().tz_localize(None) + self._server_offset

    def shutdown(self) -> None:
        if mt5 is not None and self._connected:
            mt5.shutdown()
            self._connected = False

    # --- dados --------------------------------------------------------------

    def _rates(self, timeframe_const: int, count: int, tf_minutes: int, label: str) -> pd.DataFrame:
        m = require_mt5()
        # start_pos=1 -> pula a barra em formacao
        rates = m.copy_rates_from_pos(self.symbol, timeframe_const, 1, count)
        if rates is None or len(rates) == 0:
            raise MT5Error(
                f"copy_rates_from_pos({self.symbol}, {label}) vazio: {m.last_error()}. "
                "Mercado fechado, simbolo fora do Market Watch ou contrato invalido?"
            )
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("time").sort_index()
        keep = [c for c in ("open", "high", "low", "close", "tick_volume", "real_volume", "spread") if c in df.columns]
        df = df[keep]
        # segunda trava: se por qualquer motivo a ultima barra ainda nao fechou
        return drop_forming_bar(df, tf_minutes, now=self.server_now())

    def snapshot(self, m1_bars: int = 300, m5_bars: int = 200) -> MarketSnapshot:
        m = require_mt5()
        m1 = self._rates(m.TIMEFRAME_M1, m1_bars + 2, 1, "M1")
        m5 = self._rates(m.TIMEFRAME_M5, m5_bars + 2, 5, "M5")
        tick = m.symbol_info_tick(self.symbol)
        bid = float(tick.bid) if tick and tick.bid else None
        ask = float(tick.ask) if tick and tick.ask else None
        return MarketSnapshot(
            symbol=self.symbol,
            m1=Bars("M1", m1.iloc[-m1_bars:]),
            m5=Bars("M5", m5.iloc[-m5_bars:]),
            bid=bid,
            ask=ask,
        )

    # --- diagnostico --------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Checklist para rodar ANTES de habilitar envio de ordem."""
        m = require_mt5()
        term, acct, sinfo = m.terminal_info(), m.account_info(), m.symbol_info(self.symbol)
        tick = m.symbol_info_tick(self.symbol)
        return {
            "symbol": self.symbol,
            "days_to_expiry": self._safe_dte(),
            "terminal_connected": bool(term and term.connected),
            "algo_trading_enabled": bool(term and term.trade_allowed),
            "account": getattr(acct, "login", None),
            "account_is_demo": getattr(acct, "trade_mode", None) == 0,
            "balance": getattr(acct, "balance", None),
            "symbol_trade_mode": getattr(sinfo, "trade_mode", None),
            "volume_min": getattr(sinfo, "volume_min", None),
            "volume_step": getattr(sinfo, "volume_step", None),
            "tick_size": getattr(sinfo, "trade_tick_size", None),
            "filling_mask": getattr(sinfo, "filling_mode", None),
            "bid": getattr(tick, "bid", None),
            "ask": getattr(tick, "ask", None),
            "server_offset_hours": self._server_offset.total_seconds() / 3600,
            "server_now": str(self.server_now()),
        }
