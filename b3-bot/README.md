# Robô WIN — análise 1min + 5min com IA, execução via MetaTrader 5

Robô de day trade para o mini-índice futuro da B3 (WIN). Analisa o candle de
1 minuto no contexto do gráfico de 5 minutos, decide com um de quatro
decisores (regras, ML local, LLM ou híbrido) e envia ordem pelo MetaTrader 5.

## Leia isto primeiro

**O gargalo deste projeto não é a IA.** É execução, qualidade de dado e
custo. A parte de machine learning está pronta em 300 linhas; o que separa
um robô que funciona de um que perde dinheiro são as travas de risco, a
honestidade do backtest e os 3,7 pontos de WIN que você paga só para abrir
e fechar.

**A evidência pública sobre IA operando ao vivo é fraca.** Os estudos de 2026
que avaliam agentes de LLM em operação real ("The Alpha Illusion",
LiveTradeBench) convergem: retorno reportado é quase sempre backtest, e nota
alta em benchmark de LLM não prevê resultado em trading. Por isso o decisor
`rules` existe — ele é o baseline que os outros têm que bater, líquido de
custos. Se o ML não bate uma estratégia boba, você não tem modelo: tem
overfitting.

**Day trade tem base estatística contra o varejo.** O estudo mais citado no
Brasil (Chague, De-Losso e Giannetti, FGV) encontrou que a quase totalidade
dos day traders de mini-índice que persistiram centenas de dias perdeu
dinheiro. Isso é premissa de projeto, não detalhe: seu robô precisa ser
melhor que a média por uma margem grande.

Isto é software, não recomendação de investimento. O risco é seu.

## Arquitetura

```
  MT5Feed (1min + 5min, só candle FECHADO)
        │
        ▼
  features.py ── indicadores 1min + contexto 5min + regime
        │          (sem lookahead — provado por teste)
        ▼
  decisor ─── rules │ ml │ llm │ hybrid     ← opina só sobre DIREÇÃO
        │
        ▼
  risk.py ─── sizing, stop, alvo, kill switch ← decide SE e QUANTO
        │
        ▼
  broker ─── PaperBroker │ MT5Broker
        │
        ▼
  journal.jsonl ── toda decisão auditável
```

A separação decisor/risco é o que permite trocar a IA sem tocar nas travas,
e comparar os quatro decisores sob exatamente as mesmas regras de risco.

## Os quatro decisores

| Decisor | O que é | Quando usar |
|---|---|---|
| `rules` | Rompimento com filtro de tendência no 5min | **O baseline.** Sempre rode para comparação |
| `ml` | HistGradientBoosting em 30 features | Roda em ms, determinístico, backtestável |
| `llm` | Claude analisando o contexto a cada setup | Comparação e segunda opinião |
| `hybrid` | Regras propõem, ML veta | O mais defensável: quando erra, você sabe por quê |

### Por que o LLM não entra no backtest

Latência de segundos num candle de 60 segundos, não determinismo (o mesmo
candle pode gerar respostas diferentes, então o backtest não é reproduzível)
e custo por decisão. Por isso o default é `only_on_rule_setup: true`: o LLM
só é consultado quando as regras já viram um setup. Compare-o em paper
trading ao vivo, não em backtest.

O modelo padrão é `claude-opus-5-5`. Em loop de 1 minuto a latência pesa —
`claude-sonnet-5-5` ou `claude-haiku-5-5` são mais rápidos e baratos. Essa
troca é sua decisão: meça a latência real antes.

## Instalação (Windows — obrigatório para operar)

```bat
git clone <este repo> && cd b3-bot
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```

Você também precisa:

1. **Terminal MetaTrader 5** instalado e logado numa corretora com servidor
   ligado à B3 (XP, Clear, Rico, Terra, Órama — confirme na corretora, a
   lista muda). Instalar o MT5 não cria a conexão com a B3.
2. **Botão "Algo Trading" LIGADO** no terminal. Desligado, toda ordem é
   recusada e o robô não consegue operar.
3. **Conta demo** para a fase 2. Peça à corretora.
4. `ANTHROPIC_API_KEY` apenas se for usar o decisor `llm`.

Em Linux/macOS você pode desenvolver, treinar, backtestar e rodar paper com
dados sintéticos — mas não operar. O `MetaTrader5` só existe no Windows.

## Ordem de operação

Esta ordem não é burocracia. Cada etapa pega uma classe de erro diferente.

### Fase 1 — backtest

```bash
python scripts/backtest.py --compare --source mt5 --bars 20000
```

Roda os três decisores no mesmo histórico e compara. Procure por:
`esperança/trade` positiva, `profit factor` acima de 1,3, e **custos abaixo
de 25% do bruto**. Se os custos passam de 30%, o robô opera demais.

### Fase 2 — treino do ML (se for usar `ml` ou `hybrid`)

```bash
python scripts/train_ml.py --source mt5 --bars 20000
```

A validação é **walk-forward**: cada fold treina no passado e testa no
futuro. Acurácia perto de 0,33 em 3 classes significa que o modelo não
aprendeu nada — e é o resultado honesto em dado sem sinal.

### Fase 3 — paper trading com dados reais

```yaml
# config.yaml
mode: paper
```

```bash
python scripts/run_bot.py
```

Nada é enviado. Rode por alguns dias e **audite o journal**:

```bash
python -c "import pandas as pd; d=pd.read_json('logs/journal.jsonl', lines=True); print(d.event.value_counts())"
```

Olhe cada `position_opened`: o campo `motivo` explica a entrada em
português. Se você não concorda com as decisões aqui, não avance.

### Fase 4 — conta demo

```bash
python scripts/health_check.py   # rode SEMPRE antes
```

```yaml
mode: demo
```

Agora a ordem sai de verdade, com dinheiro fictício. Esta fase pega o que o
paper não pega: rejeição de ordem, preenchimento parcial, slippage real,
modo de preenchimento errado. Compare o `price_hint` com o `entry` no
journal — a diferença é seu slippage real.

### Fase 5 — conta real

```yaml
mode: live
live_confirmation: "EU ASSUMO O RISCO"
```

Sem a frase exata, o robô se recusa a subir. Comece com
`max_contracts: 1` e `max_daily_loss_brl` no valor que você aceita perder
num dia ruim, sem hesitar.

## Travas de segurança

| Trava | O que faz |
|---|---|
| **Kill switch diário** | Atinge `max_daily_loss_brl` → o dia acaba. Conta o PnL **em aberto** |
| **Perdas consecutivas** | N perdas seguidas → para o dia |
| **Stop obrigatório** | Sem stop calculável, não opera |
| **Zeragem forçada** | Zera antes do fim do pregão. Nunca carrega overnight |
| **R:R líquido mínimo** | Rejeita setup cujo payoff não sobrevive ao custo |
| **Reconciliação** | Posição divergente do broker → **para** |
| **Feed travado** | Dado velho com posição aberta → zera e para |
| **Teto de volatilidade** | ATR alto demais para o seu stop → não opera |
| **Cooldown** | N candles de espera depois de sair |
| **Trava de modo** | `live` exige frase de confirmação explícita |

## O que confirmar antes de operar

- [ ] **Emolumentos** na tabela vigente da B3 e com sua corretora. Variam
      com seu volume médio diário. O default (R$ 0,27/contrato/ponta) é
      conservador, não oficial.
- [ ] **Valor do ponto do WIN** (R$ 0,20) e **tick** (5 pontos).
- [ ] **Código do contrato** no MT5 da sua corretora. O robô resolve por
      vencimento (`WINZ26`) e confere contra os símbolos reais, mas algumas
      corretoras usam sufixos.
- [ ] **Horário do servidor MT5.** Frequentemente não é horário de Brasília.
      O robô mede o offset, mas confira no health check.
- [ ] **Tributação**: day trade 20%, IRRF 1% na fonte é antecipação, DARF
      6015. Não existe isenção de R$ 20 mil em day trade.

## Testes

```bash
python -m pytest tests/ -q     # 60 testes
```

Os mais importantes são `tests/test_no_lookahead.py`. Eles provam
empiricamente que as features não olham o futuro: calculam tudo sobre a
série completa e sobre a série truncada no meio, e exigem que os valores do
passado sejam idênticos. **Se algum falhar, todo backtest do projeto está
invalidado** — conserte a feature, não o teste.

## Limitações conhecidas

O que este robô **não** modela, e você precisa saber:

- **Fila em ordem limitada.** O backtest assume que o alvo executa quando o
  preço toca. Na fila real você pode não ser atendido.
- **Liquidez no seu tamanho.** Com 1 contrato em WIN isso é desprezível.
  Com 20, não é.
- **Book de ofertas.** Nenhuma feature de microestrutura (desequilíbrio de
  book, agressão, fluxo). Histórico de negócios da B3 não inclui o book —
  se quiser isso, comece a gravar hoje, porque não se compra esse passado.
- **Gap intradiário** e leilões.
- **Feriados da B3** no cálculo de vencimento (a folga de rolagem cobre, mas
  não é um calendário).

## Estrutura

```
b3bot/
  config.py      config validada; trava do modo live
  contracts.py   resolução do contrato por vencimento
  costs.py       emolumentos, registro, IR de day trade
  features.py    indicadores, regime, sincronização 1min/5min
  risk.py        sizing, stop, kill switch
  engine.py      o loop
  backtest.py    motor de backtest
  journal.py     auditoria em JSONL
  data/          MT5Feed (real) e SyntheticFeed (teste)
  decision/      rules, ml_local, llm, hybrid
  execution/     PaperBroker e MT5Broker
scripts/
  health_check.py   checklist antes de operar
  run_bot.py        sobe o robô (modo vem do config)
  backtest.py       backtest e comparativo
  train_ml.py       treino walk-forward
tests/              60 testes, incl. provas de ausência de lookahead
```
