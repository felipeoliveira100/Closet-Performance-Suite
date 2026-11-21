"""Testes de integracao do engine: estado, reconciliacao e kill switch.

E aqui que moram os bugs que custam dinheiro de verdade - nao na
estrategia, mas no robo achando que tem posicao quando nao tem.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pytest

from b3bot.data.synthetic import SyntheticFeed
from b3bot.decision.base import Decisor, Signal, hold
from b3bot.engine import TradingEngine
from b3bot.execution.base import Broker, BrokerPosition, OrderResult
from b3bot.execution.paper import PaperBroker
from b3bot.journal import Journal


class AlwaysBuy(Decisor):
    name = "always_buy"

    def warmup_bars(self) -> int:
        return 60

    def decide(self, snap) -> Signal:
        return Signal("BUY", 0.9, "teste: compra sempre", self.name)


class AlwaysHold(Decisor):
    name = "always_hold"

    def warmup_bars(self) -> int:
        return 60

    def decide(self, snap) -> Signal:
        return hold(self.name, "teste: nunca opera")


class BrokenBroker(Broker):
    """Broker que aceita abrir e falha ao zerar - o pior caso operacional."""
    name = "broken"
    sends_real_orders = False

    def __init__(self):
        self._pos = None

    def connect(self): pass
    def shutdown(self): pass

    def market_order(self, side, contracts, price_hint, comment=""):
        self._pos = BrokerPosition(side, contracts, price_hint)
        return OrderResult(True, side, contracts, price_hint, retcode=0, order_id=1)

    def close_all(self, price_hint, comment=""):
        return OrderResult(False, retcode=10013, comment="recusada pelo servidor")

    def current_position(self):
        return self._pos


def make_engine(cfg, decisor, broker=None, tmp_path=None, n_bars=900, seed=3):
    feed = SyntheticFeed(n_bars=n_bars, seed=seed, warmup=120)
    broker = broker or PaperBroker(cfg)
    broker.connect()
    journal = Journal((tmp_path or "/tmp") / "j.jsonl" if tmp_path else "/tmp/j_test.jsonl",
                      mode="paper")
    return TradingEngine(cfg, feed, broker, decisor, journal), feed, broker


def test_abre_posicao_com_sinal(cfg, tmp_path):
    eng, feed, _ = make_engine(cfg, AlwaysBuy(), tmp_path=tmp_path)
    for _ in range(40):
        eng.step(now=feed.now.to_pydatetime())
        if eng.position is not None:
            break
        feed.advance(1)
    assert eng.position is not None, "engine nao abriu posicao mesmo com sinal constante"
    assert eng.position.side == "BUY"
    assert eng.position.stop_price < eng.position.entry_price < eng.position.target_price


def test_nao_processa_a_mesma_barra_duas_vezes(cfg, tmp_path):
    """Timer cego processaria a barra de novo e mandaria ordem duplicada.

    Vale quando o robo esta FLAT. Com posicao aberta o step processa de novo
    de proposito, porque stop/alvo/fim de pregao dependem do relogio.
    """
    eng, feed, _ = make_engine(cfg, AlwaysHold(), tmp_path=tmp_path)
    eng.step(now=feed.now.to_pydatetime())
    r2 = eng.step(now=feed.now.to_pydatetime())
    assert r2.reason == "nenhuma barra nova"
    assert not r2.acted


def test_com_posicao_aberta_reprocessa_a_mesma_barra(cfg, tmp_path):
    """O oposto do teste acima, e igualmente importante: o relogio anda
    mesmo sem barra nova, e o fim de pregao nao pode ser perdido."""
    eng, feed, _ = make_engine(cfg, AlwaysBuy(), tmp_path=tmp_path)
    for _ in range(40):
        eng.step(now=feed.now.to_pydatetime())
        if eng.position is not None:
            break
        feed.advance(1)
    assert eng.position is not None
    r = eng.step(now=feed.now.to_pydatetime())  # mesma barra
    assert r.reason != "nenhuma barra nova"


def test_kill_switch_impede_novas_entradas(cfg, tmp_path):
    eng, feed, _ = make_engine(cfg, AlwaysBuy(), tmp_path=tmp_path)
    # primeiro step alinha o dia interno com o do feed; so depois paramos,
    # porque a virada de dia reseta o kill switch (de proposito: ele e diario)
    eng.step(now=feed.now.to_pydatetime())
    eng.risk.halt("teste")
    for _ in range(30):
        feed.advance(1)
        eng.step(now=feed.now.to_pydatetime())
    assert eng.position is None, "abriu posicao com o robo parado"


def test_reconciliacao_detecta_posicao_desconhecida(cfg, tmp_path):
    """Broker tem posicao que o robo nao conhece (ordem manual, stop da corretora)."""
    eng, feed, broker = make_engine(cfg, AlwaysHold(), tmp_path=tmp_path)
    broker._pos = BrokerPosition("BUY", 1, 140000, ticket=99)
    res = eng.step(now=feed.now.to_pydatetime())
    assert eng.risk.state.halted
    assert "nao conhece" in res.reason


def test_reconciliacao_detecta_posicao_sumida(cfg, tmp_path):
    """Robo acha que esta posicionado, broker nao tem nada."""
    eng, feed, broker = make_engine(cfg, AlwaysBuy(), tmp_path=tmp_path)
    for _ in range(40):
        eng.step(now=feed.now.to_pydatetime())
        if eng.position is not None:
            break
        feed.advance(1)
    assert eng.position is not None
    broker._pos = None  # alguem zerou por fora
    feed.advance(1)
    res = eng.step(now=feed.now.to_pydatetime())
    assert eng.risk.state.halted
    assert "broker nao tem posicao" in res.reason


def test_reconciliacao_detecta_quantidade_divergente(cfg, tmp_path):
    eng, feed, broker = make_engine(cfg, AlwaysBuy(), tmp_path=tmp_path)
    for _ in range(40):
        eng.step(now=feed.now.to_pydatetime())
        if eng.position is not None:
            break
        feed.advance(1)
    assert eng.position is not None
    p = broker._pos
    broker._pos = BrokerPosition(p.side, p.contracts + 3, p.entry_price, ticket=p.ticket)
    feed.advance(1)
    res = eng.step(now=feed.now.to_pydatetime())
    assert eng.risk.state.halted and "divergente" in res.reason


def test_falha_ao_zerar_nao_apaga_a_posicao_interna(cfg, tmp_path):
    """Se a ordem de saida falha, o robo NAO pode esquecer que esta exposto."""
    eng, feed, _ = make_engine(cfg, AlwaysBuy(), BrokenBroker(), tmp_path=tmp_path)
    for _ in range(40):
        eng.step(now=feed.now.to_pydatetime())
        if eng.position is not None:
            break
        feed.advance(1)
    assert eng.position is not None
    pos = eng.position
    # Forca saida por STOP: coloca o stop acima do preco atual, de modo que a
    # proxima barra o tenha tocado com certeza.
    eng.position.stop_price = eng.position.entry_price + 10_000
    feed.advance(1)
    res = eng.step(now=feed.now.to_pydatetime())
    assert eng.position is pos, "posicao interna apagada apesar da falha ao zerar"
    assert "FALHA ao zerar" in res.reason


def test_shutdown_zera_posicao_aberta(cfg, tmp_path):
    eng, feed, broker = make_engine(cfg, AlwaysBuy(), tmp_path=tmp_path)
    for _ in range(40):
        eng.step(now=feed.now.to_pydatetime())
        if eng.position is not None:
            break
        feed.advance(1)
    assert eng.position is not None
    eng.shutdown()
    assert eng.position is None, "shutdown deixou posicao aberta"
    assert broker.current_position() is None


def test_decisor_que_explode_nao_derruba_o_engine(cfg, tmp_path):
    class Explode(Decisor):
        name = "explode"
        def warmup_bars(self): return 60
        def decide(self, snap):
            raise RuntimeError("boom")

    eng, feed, _ = make_engine(cfg, Explode(), tmp_path=tmp_path)
    for _ in range(20):
        res = eng.step(now=feed.now.to_pydatetime())  # nao deve levantar
        feed.advance(1)
    assert eng.position is None


def test_modo_paper_nunca_marca_ordens_reais(cfg, tmp_path):
    eng, feed, broker = make_engine(cfg, AlwaysBuy(), tmp_path=tmp_path)
    assert broker.sends_real_orders is False
    assert cfg.mode == "paper"
    assert cfg.sends_real_orders is False


def test_feed_travado_com_posicao_aberta_zera_e_para(cfg, tmp_path):
    """Bug real encontrado durante os testes: a checagem de dado velho rodava
    ANTES da gestao da posicao, deixando risco aberto sem ninguem olhando o
    stop. O fail-safe correto e zerar a mercado e parar o dia."""
    eng, feed, broker = make_engine(cfg, AlwaysBuy(), tmp_path=tmp_path)
    for _ in range(40):
        eng.step(now=feed.now.to_pydatetime())
        if eng.position is not None:
            break
        feed.advance(1)
    assert eng.position is not None

    # simula feed congelado: o relogio anda horas, a barra nao
    futuro = feed.now.to_pydatetime().replace(hour=feed.now.hour) + __import__("datetime").timedelta(hours=3)
    res = eng.step(now=futuro)
    assert eng.position is None, "ficou exposto com feed travado"
    assert eng.risk.state.halted
    assert broker.current_position() is None
    assert "feed velho" in res.reason
