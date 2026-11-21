"""Decisor 3 de 3: LLM (Claude) analisando o contexto de mercado.

LEIA ANTES DE USAR EM CONTA REAL. Este decisor existe porque voce pediu
para comparar os tres, e a comparacao e informativa. Mas ele tem tres
limitacoes que nao sao de implementacao - sao da natureza da abordagem:

  1) LATENCIA. Um candle de 1 minuto da 60 segundos de janela. A chamada
     leva alguns segundos. Funciona, mas o preco se move durante a
     inferencia, e a decisao chega sobre um preco que mudou.
  2) NAO DETERMINISMO. O mesmo candle pode gerar respostas diferentes.
     Isso torna o backtest nao reproduzivel: voce nao consegue provar que
     a estrategia funcionava, so que funcionou naquela execucao.
  3) CUSTO POR DECISAO. Por isso o default e `only_on_rule_setup: true`:
     o LLM e consultado apenas quando as regras ja viram um setup,
     atuando como segunda opiniao, nao como varredor de todo candle.

A evidencia publica de 2026 sobre agentes de LLM operando ao vivo
("The Alpha Illusion", LiveTradeBench) e desfavoravel: retorno reportado
costuma ser backtest, e nota alta em benchmark de LLM nao prevê resultado
em trading. Trate este decisor como instrumento de comparacao.

Requer: pip install anthropic, e a variavel ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from ..features import FeatureSnapshot
from .base import Decisor, Signal, hold
from .rules import RulesDecisor

logger = logging.getLogger(__name__)

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore

# Schema da resposta. output_config.format garante JSON valido e com estas
# chaves - evita parsear texto livre do modelo.
DECISION_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reason": {"type": "string", "maxLength": 300},
            "key_risk": {"type": "string", "maxLength": 200},
        },
        "required": ["action", "confidence", "reason", "key_risk"],
        "additionalProperties": False,
    },
}

# System prompt ESTAVEL byte a byte: e o que permite o cache de prompt
# funcionar e derrubar o custo das chamadas seguintes.
SYSTEM_PROMPT = """Voce analisa o mini-indice futuro da B3 (WIN) em day trade intradiario.

Recebe o estado do ultimo candle de 1 minuto JA FECHADO, mais o contexto do
grafico de 5 minutos. Responde se abre posicao agora.

Regras do seu papel:
- Voce opina SOMENTE sobre direcao e convicção. Tamanho de posicao, stop e
  alvo sao definidos por um modulo de risco separado - nao sugira valores.
- As distancias vem normalizadas em ATR (ex: "+0.8 atr acima do VWAP"), para
  serem comparaveis entre niveis de preco e de volatilidade.
- HOLD e a resposta correta na duvida. Operar sem vantagem clara perde para
  os custos. Um robo que opera pouco e melhor que um que opera sempre.
- Day trade de indice: tendencia definida no 5 minutos favorece continuacao;
  mercado lateral favorece reversao a media; volatilidade no topo da
  distribuicao recente (high_vol) pede cautela, porque o stop normal nao
  aguenta o ruido.
- confidence e sua convicção real, nao entusiasmo. Acima de 0.8 so com
  varios fatores alinhados.
- key_risk: em uma frase, o que invalida sua leitura.

Nao de conselho de investimento, nao prometa retorno, nao mencione que e uma IA."""


def _fmt(v: float, nd: int = 2) -> str:
    try:
        f = float(v)
        if f != f:  # NaN
            return "n/d"
        return f"{f:+.{nd}f}"
    except (TypeError, ValueError):
        return "n/d"


def build_market_brief(snap: FeatureSnapshot) -> str:
    """Estado do mercado em texto compacto.

    Texto em vez de imagem do grafico: mais rapido, mais barato e, acima de
    tudo, reproduzivel. Uma imagem renderizada acrescenta latencia de
    renderizacao e upload a um loop que ja e apertado.
    """
    g = snap.get
    return f"""HORARIO: {snap.time:%d/%m/%Y %H:%M} (candle de 1min fechado)
PRECO: {snap.price:.0f} pontos | ATR(14) 1min: {snap.atr:.0f} pontos
REGIME NO 5 MINUTOS: {snap.regime}

CANDLE DE 1 MINUTO (tudo normalizado em ATR):
  distancia da EMA9:      {_fmt(g('dist_ema_fast_atr'))} atr
  distancia da EMA21:     {_fmt(g('dist_ema_slow_atr'))} atr
  spread EMA9-EMA21:      {_fmt(g('ema_spread_atr'))} atr
  distancia do VWAP:      {_fmt(g('dist_vwap_atr'))} atr
  rompimento de maxima:   {_fmt(g('breakout_up_atr'))} atr (positivo = rompeu)
  rompimento de minima:   {_fmt(g('breakout_dn_atr'))} atr (positivo = rompeu)
  posicao nas Bollinger:  {_fmt(g('bb_pos'))} (0=banda inf, 1=banda sup)
  RSI(14):                {g('rsi'):.0f}
  retorno 5 candles:      {_fmt(g('ret_5_atr'))} atr
  retorno 15 candles:     {_fmt(g('ret_15_atr'))} atr
  volume vs media 20:     {_fmt(g('vol_ratio'))}x
  corpo do candle:        {_fmt(g('body_ratio'))} da amplitude
  pavio superior:         {_fmt(g('upper_wick_ratio'))} | inferior: {_fmt(g('lower_wick_ratio'))}
  padroes:                engolfo_alta={int(g('bull_engulf', 0))} engolfo_baixa={int(g('bear_engulf', 0))} martelo={int(g('hammer', 0))} estrela={int(g('shooting_star', 0))}

CONTEXTO DO 5 MINUTOS:
  inclinacao da EMA21:    {_fmt(g('m5_slope_atr'))} atr
  distancia da EMA21:     {_fmt(g('m5_dist_ema21_atr'))} atr
  spread EMA9-EMA21:      {_fmt(g('m5_ema_spread_atr'))} atr
  RSI(14):                {g('m5_rsi'):.0f}
  percentil do ATR:       {_fmt(g('m5_atr_rank'))} (1.0 = vol maxima recente)
  posicao no range de 12: {_fmt(g('m5_pos_in_range'))}

Abro posicao agora?"""


class LLMDecisor(Decisor):
    name = "llm"

    def __init__(self, params: dict[str, Any] | None = None,
                 allow_long: bool = True, allow_short: bool = False,
                 rules_params: dict[str, Any] | None = None,
                 client: Any | None = None) -> None:
        p = params or {}
        self.model = str(p.get("model", "claude-opus-5-5"))
        self.max_tokens = int(p.get("max_tokens", 700))
        self.min_confidence = float(p.get("min_confidence", 0.60))
        self.only_on_rule_setup = bool(p.get("only_on_rule_setup", True))
        self.timeout_seconds = float(p.get("timeout_seconds", 20))
        self.effort = str(p.get("effort", "low"))  # latencia importa no loop de 1min
        self.allow_long = allow_long
        self.allow_short = allow_short
        self._gate = RulesDecisor(rules_params, allow_long, allow_short)
        self.calls = 0
        self.errors = 0

        self._client = client
        self._init_error: str | None = None
        if self._client is None:
            if anthropic is None:
                self._init_error = "pacote anthropic nao instalado (pip install anthropic)"
            elif not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
                self._init_error = "ANTHROPIC_API_KEY nao definida no ambiente"
            else:
                try:
                    self._client = anthropic.Anthropic()
                except Exception as exc:  # noqa: BLE001
                    self._init_error = f"falha ao criar client: {exc}"
        if self._init_error:
            logger.error("LLMDecisor indisponivel: %s", self._init_error)

    @property
    def ready(self) -> bool:
        return self._client is not None

    def warmup_bars(self) -> int:
        return max(60, self._gate.warmup_bars())

    def decide(self, snap: FeatureSnapshot) -> Signal:
        if not self.ready:
            return hold(self.name, f"LLM indisponivel: {self._init_error}")

        # Portao economico: so consulta o LLM quando ha setup pelas regras.
        if self.only_on_rule_setup:
            gate = self._gate.safe_decide(snap)
            if not gate.is_entry:
                return hold(self.name, f"sem setup pelas regras ({gate.reason}); LLM nao consultado")

        brief = build_market_brief(snap)
        try:
            self.calls += 1
            response = self._client.with_options(
                timeout=self.timeout_seconds, max_retries=1
            ).messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                output_config={"effort": self.effort, "format": DECISION_SCHEMA},
                messages=[{"role": "user", "content": brief}],
            )
        except Exception as exc:  # noqa: BLE001
            self.errors += 1
            # Falha de rede/API NUNCA vira entrada. HOLD e o fail-safe.
            return hold(self.name, f"falha na chamada ({type(exc).__name__}): {exc}")

        if getattr(response, "stop_reason", None) == "refusal":
            return hold(self.name, "modelo recusou responder")

        text = next((b.text for b in response.content if getattr(b, "type", "") == "text"), None)
        if not text:
            return hold(self.name, "resposta sem bloco de texto")

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            return hold(self.name, f"JSON invalido na resposta: {exc}")

        action = str(data.get("action", "HOLD")).upper()
        conf = float(data.get("confidence", 0.0))
        reason = str(data.get("reason", ""))[:300]
        key_risk = str(data.get("key_risk", ""))[:200]

        usage = getattr(response, "usage", None)
        meta = {
            "key_risk": key_risk,
            "model": self.model,
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "cache_read_tokens": getattr(usage, "cache_read_input_tokens", None),
        }

        if action not in ("BUY", "SELL", "HOLD"):
            return hold(self.name, f"action invalida: {action!r}", **meta)
        if action == "BUY" and not self.allow_long:
            return hold(self.name, "compra desabilitada na config", **meta)
        if action == "SELL" and not self.allow_short:
            return hold(self.name, "venda desabilitada na config", **meta)
        if action != "HOLD" and conf < self.min_confidence:
            return hold(self.name,
                        f"confianca {conf:.2f} < minimo {self.min_confidence:.2f}: {reason}",
                        **meta)
        if action == "HOLD":
            return hold(self.name, reason or "LLM optou por aguardar", **meta)

        return Signal(action, min(1.0, max(0.0, conf)), reason, self.name, meta)
