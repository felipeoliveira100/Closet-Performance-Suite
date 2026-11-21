"""Decisor 2 de 3: modelo de ML local (HistGradientBoosting).

Por que este e o tipo de IA que eu recomendo para operar de verdade:
roda em milissegundos, e deterministico e - o que mais importa - pode ser
backtestado honestamente. Um LLM opinando a cada candle nao tem nenhuma
das tres propriedades.

ONDE O LOOKAHEAD E LEGITIMO: no ROTULO, nunca na feature. O alvo de
treino e "o preco subiu nos proximos N candles?", o que obviamente olha o
futuro - e por isso que ele e um rotulo e nao um input. A funcao
make_labels usa shift negativo de proposito e por isso vive aqui, e nao
em features.py (onde o teste por AST proibe shift negativo).

VALIDACAO: split SEMPRE temporal, nunca aleatorio. train_test_split
embaralhado em serie temporal treina com o futuro e testa no passado -
produz acuracia alta e prejuizo real.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..features import ML_FEATURE_COLS, FeatureSnapshot
from .base import Decisor, Signal, hold

logger = logging.getLogger(__name__)

CLASS_DOWN, CLASS_FLAT, CLASS_UP = -1, 0, 1


def make_labels(frame: pd.DataFrame, horizon_bars: int = 5,
                threshold_atr: float = 0.5) -> pd.Series:
    """Rotulo: o preco andou mais de `threshold_atr` ATRs em `horizon` candles?

    shift(-horizon) aqui e CORRETO e necessario - e o alvo supervisionado.
    As ultimas `horizon` barras ficam sem rotulo (NaN) e sao descartadas
    no treino.
    """
    future_close = frame["close"].shift(-horizon_bars)
    move_atr = (future_close - frame["close"]) / frame["atr"].replace(0.0, np.nan)
    labels = pd.Series(CLASS_FLAT, index=frame.index, dtype="float")
    labels[move_atr > threshold_atr] = CLASS_UP
    labels[move_atr < -threshold_atr] = CLASS_DOWN
    labels[move_atr.isna()] = np.nan
    return labels


@dataclass
class TrainReport:
    n_train: int
    n_test: int
    folds: list[dict[str, Any]] = field(default_factory=list)
    class_balance: dict[str, float] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"treino: {self.n_train} barras | teste: {self.n_test} barras"]
        lines.append(f"balanco de classes: {self.class_balance}")
        for f in self.folds:
            lines.append(
                f"  fold {f['fold']}: acuracia {f['accuracy']:.3f} | "
                f"precisao_UP {f['precision_up']:.3f} | recall_UP {f['recall_up']:.3f} | "
                f"n={f['n_test']}"
            )
        if self.folds:
            accs = [f["accuracy"] for f in self.folds]
            precs = [f["precision_up"] for f in self.folds]
            lines.append(
                f"  media: acuracia {np.mean(accs):.3f} | precisao_UP {np.mean(precs):.3f}"
            )
        lines.append(
            "  AVISO: acuracia nao e lucro. Um modelo com 55% de acerto e "
            "payoff ruim perde dinheiro. Julgue pelo backtest liquido de custos."
        )
        return "\n".join(lines)


def walk_forward_train(frame: pd.DataFrame, horizon_bars: int = 5,
                       threshold_atr: float = 0.5, n_folds: int = 4,
                       min_train: int = 500, random_state: int = 0
                       ) -> tuple[Any, TrainReport]:
    """Treina com validacao walk-forward e devolve o modelo final.

    Walk-forward: fold k treina em [0, t_k) e testa em [t_k, t_k+1).
    O teste esta SEMPRE no futuro do treino, como na vida real.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    cols = [c for c in ML_FEATURE_COLS if c in frame.columns]
    if not cols:
        raise ValueError("nenhuma feature de ML presente no frame")

    labels = make_labels(frame, horizon_bars, threshold_atr)
    data = frame[cols].copy()
    data["_y"] = labels
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["_y"])
    # features podem ter NaN: HistGradientBoosting lida com isso nativamente
    data = data[data[cols].notna().sum(axis=1) >= max(1, len(cols) // 2)]

    if len(data) < min_train + 100:
        raise ValueError(
            f"dados insuficientes: {len(data)} barras utilizaveis, "
            f"minimo {min_train + 100}. Precisa de mais historico."
        )

    X = data[cols].to_numpy(dtype=float)
    y = data["_y"].to_numpy(dtype=int)

    report = TrainReport(n_train=0, n_test=0)
    vals, counts = np.unique(y, return_counts=True)
    names = {CLASS_DOWN: "DOWN", CLASS_FLAT: "FLAT", CLASS_UP: "UP"}
    report.class_balance = {names.get(int(v), str(v)): round(c / len(y), 3)
                            for v, c in zip(vals, counts)}

    fold_size = (len(X) - min_train) // n_folds
    if fold_size < 50:
        n_folds = max(1, (len(X) - min_train) // 50)
        fold_size = (len(X) - min_train) // max(1, n_folds)

    for k in range(n_folds):
        tr_end = min_train + k * fold_size
        te_end = tr_end + fold_size if k < n_folds - 1 else len(X)
        if te_end - tr_end < 20:
            continue
        model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, max_depth=4,
            min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15,
            random_state=random_state,
        )
        model.fit(X[:tr_end], y[:tr_end])
        pred = model.predict(X[tr_end:te_end])
        truth = y[tr_end:te_end]
        up_pred = pred == CLASS_UP
        report.folds.append({
            "fold": k + 1,
            "n_train": int(tr_end),
            "n_test": int(te_end - tr_end),
            "accuracy": float((pred == truth).mean()),
            "precision_up": float((truth[up_pred] == CLASS_UP).mean()) if up_pred.any() else 0.0,
            "recall_up": float(up_pred[truth == CLASS_UP].mean()) if (truth == CLASS_UP).any() else 0.0,
            "n_up_predicted": int(up_pred.sum()),
        })

    # modelo final: treinado em tudo, para operar daqui para frente
    final = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.06, max_depth=4,
        min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15,
        random_state=random_state,
    )
    final.fit(X, y)
    report.n_train = len(X)
    report.n_test = sum(f["n_test"] for f in report.folds)

    bundle = {
        "model": final,
        "feature_cols": cols,
        "classes": list(final.classes_),
        "horizon_bars": horizon_bars,
        "threshold_atr": threshold_atr,
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "n_samples": len(X),
    }
    return bundle, report


def save_bundle(bundle: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(bundle, fh)
    logger.info("modelo salvo em %s (%d amostras)", path, bundle.get("n_samples", 0))


def load_bundle(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"modelo nao encontrado: {path}\n"
            "  Treine primeiro: python scripts/train_ml.py"
        )
    with path.open("rb") as fh:
        return pickle.load(fh)


class MLDecisor(Decisor):
    name = "ml"

    def __init__(self, params: dict[str, Any] | None = None,
                 allow_long: bool = True, allow_short: bool = False,
                 bundle: dict[str, Any] | None = None) -> None:
        p = params or {}
        self.min_confidence = float(p.get("min_confidence", 0.58))
        self.model_path = p.get("model_path", "models/win_m1.pkl")
        self.allow_long = allow_long
        self.allow_short = allow_short
        self._bundle = bundle
        self._load_error: str | None = None
        if self._bundle is None:
            try:
                self._bundle = load_bundle(self.model_path)
            except Exception as exc:  # noqa: BLE001
                self._load_error = str(exc)
                logger.error("MLDecisor sem modelo: %s", exc)

    @property
    def ready(self) -> bool:
        return self._bundle is not None

    def warmup_bars(self) -> int:
        return 120

    def decide(self, snap: FeatureSnapshot) -> Signal:
        if not self.ready:
            return hold(self.name, f"modelo nao carregado: {self._load_error}")

        cols = self._bundle["feature_cols"]
        row = np.array([[snap.get(c) for c in cols]], dtype=float)
        row[~np.isfinite(row)] = np.nan
        if np.isnan(row).all():
            return hold(self.name, "todas as features indisponiveis (warmup)")

        model = self._bundle["model"]
        proba = model.predict_proba(row)[0]
        classes = list(model.classes_)

        def p(cls: int) -> float:
            return float(proba[classes.index(cls)]) if cls in classes else 0.0

        p_up, p_down, p_flat = p(CLASS_UP), p(CLASS_DOWN), p(CLASS_FLAT)
        meta = {"p_up": round(p_up, 4), "p_down": round(p_down, 4),
                "p_flat": round(p_flat, 4), "regime": snap.regime}

        if self.allow_long and p_up >= self.min_confidence and p_up > p_down:
            return Signal("BUY", p_up,
                          f"modelo: P(alta)={p_up:.2f} vs P(baixa)={p_down:.2f} "
                          f"em {self._bundle['horizon_bars']} candles",
                          self.name, meta)
        if self.allow_short and p_down >= self.min_confidence and p_down > p_up:
            return Signal("SELL", p_down,
                          f"modelo: P(baixa)={p_down:.2f} vs P(alta)={p_up:.2f}",
                          self.name, meta)

        melhor = max(p_up, p_down)
        return hold(self.name,
                    f"confianca {melhor:.2f} < minimo {self.min_confidence:.2f} "
                    f"(alta {p_up:.2f} / baixa {p_down:.2f})", **meta)

    def score_up(self, snap: FeatureSnapshot) -> float:
        """Probabilidade de alta - usada pelo decisor hibrido como filtro."""
        if not self.ready:
            return float("nan")
        cols = self._bundle["feature_cols"]
        row = np.array([[snap.get(c) for c in cols]], dtype=float)
        row[~np.isfinite(row)] = np.nan
        model = self._bundle["model"]
        proba = model.predict_proba(row)[0]
        classes = list(model.classes_)
        return float(proba[classes.index(CLASS_UP)]) if CLASS_UP in classes else float("nan")
