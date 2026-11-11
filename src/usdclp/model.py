"""Modelos de probabilidad y validacion walk-forward con embargo.

Se combinan dos modelos complementarios:

* Regresion logistica con penalizacion L2 sobre variables estandarizadas:
  estable, interpretable, bien calibrada por construccion.
* Gradient boosting (LightGBM) poco profundo y muy regularizado: capta
  no linealidades e interacciones (p. ej. el efecto de los pares solo
  importa cuando la volatilidad esta alta).

La probabilidad final es el promedio de ambos, recalibrado con una regresion
logistica de un parametro (Platt) ajustada solo con predicciones fuera de
muestra de anos anteriores.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


def make_logit(C: float = 0.01):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2000),
    )


def make_gbm(seed: int = config.RANDOM_STATE, **kw):
    params = dict(
        n_estimators=300,
        learning_rate=0.02,
        num_leaves=7,
        max_depth=3,
        min_child_samples=150,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_lambda=5.0,
        random_state=seed,
        verbose=-1,
    )
    params.update(kw)
    return lgb.LGBMClassifier(**params)


def clip_features(X: pd.DataFrame, lo: pd.Series, hi: pd.Series) -> pd.DataFrame:
    """Winsoriza con percentiles del set de entrenamiento (robustez a outliers)."""
    return X.clip(lower=lo, upper=hi, axis=1)


@dataclass
class Ensemble:
    """Logit + GBM. Se ajusta y predice como un clasificador sklearn."""

    features: list[str]
    C: float = 0.01
    use_gbm: bool = True
    gbm_kw: dict = field(default_factory=dict)
    n_seeds: int = 3

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None):
        X = X[self.features]
        self.lo_ = X.quantile(0.005)
        self.hi_ = X.quantile(0.995)
        Xc = clip_features(X, self.lo_, self.hi_)
        self.logit_ = make_logit(self.C).fit(Xc, y, logisticregression__sample_weight=sample_weight)
        self.gbms_ = []
        if self.use_gbm and lgb is not None:
            for s in range(self.n_seeds):
                self.gbms_.append(make_gbm(seed=config.RANDOM_STATE + s, **self.gbm_kw).fit(Xc, y, sample_weight=sample_weight))
        return self

    def predict_components(self, X: pd.DataFrame) -> pd.DataFrame:
        Xc = clip_features(X[self.features], self.lo_, self.hi_)
        out = pd.DataFrame(index=X.index)
        out["p_logit"] = self.logit_.predict_proba(Xc)[:, 1]
        if self.gbms_:
            out["p_gbm"] = np.mean([g.predict_proba(Xc)[:, 1] for g in self.gbms_], axis=0)
        return out

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        comp = self.predict_components(X)
        return comp.mean(axis=1).values


def make_ridge(alpha: float = 300.0):
    from sklearn.linear_model import Ridge
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))


def vol_scaled_target(aux: pd.DataFrame, clip: float = 4.0) -> pd.Series:
    """Exceso de t a t+1 dividido por la volatilidad diaria reciente y acotado.

    Normalizar por volatilidad hace comparables jornadas de 2008 y de 2017, y
    acotar evita que un par de dias extremos dominen el ajuste."""
    sd = aux["clp_rv20"] / np.sqrt(252)
    return (aux["excess_next"] / sd).clip(-clip, clip)


@dataclass
class RegEnsemble:
    """Ridge + LightGBM de regresion sobre el retorno normalizado por vol.

    A diferencia del clasificador, usa la magnitud del movimiento: un dia de
    +2 desvios pesa mas que uno de +0,1. La prediccion (en desvios) se lleva a
    una pseudo-probabilidad sigmoid(z) que luego calibra el mismo temperature
    scaling walk-forward."""

    features: list[str]
    alpha: float = 300.0
    n_seeds: int = 3

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None, aux: pd.DataFrame | None = None):
        X = X[self.features]
        z = vol_scaled_target(aux)
        ok = z.notna()
        X, z = X[ok], z[ok]
        w = None if sample_weight is None else np.asarray(sample_weight)[ok.values]
        self.lo_, self.hi_ = X.quantile(0.005), X.quantile(0.995)
        Xc = clip_features(X, self.lo_, self.hi_)
        self.ridge_ = make_ridge(self.alpha).fit(Xc, z, ridge__sample_weight=w)
        self.gbms_ = []
        if lgb is not None:
            for s in range(self.n_seeds):
                prm = dict(n_estimators=300, learning_rate=0.02, num_leaves=7, max_depth=3,
                           min_child_samples=150, subsample=0.7, subsample_freq=1,
                           colsample_bytree=0.5, reg_lambda=5.0, random_state=config.RANDOM_STATE + s,
                           verbose=-1, objective="huber", alpha=1.5)
                self.gbms_.append(lgb.LGBMRegressor(**prm).fit(Xc, z, sample_weight=w))
        return self

    def predict_components(self, X: pd.DataFrame) -> pd.DataFrame:
        Xc = clip_features(X[self.features], self.lo_, self.hi_)
        out = pd.DataFrame(index=X.index)
        out["p_ridge"] = 1 / (1 + np.exp(-self.ridge_.predict(Xc)))
        if self.gbms_:
            zg = np.mean([g.predict(Xc) for g in self.gbms_], axis=0)
            out["p_gbmr"] = 1 / (1 + np.exp(-zg))
        return out

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_components(X).mean(axis=1).values


@dataclass
class Combo:
    """Promedio del clasificador (direccion) y del regresor (magnitud)."""

    features: list[str]
    C: float = 0.005
    alpha: float = 300.0
    n_seeds: int = 3

    def fit(self, X, y, sample_weight=None, aux=None):
        self.clf_ = Ensemble(self.features, C=self.C, n_seeds=self.n_seeds).fit(X, y, sample_weight)
        self.reg_ = RegEnsemble(self.features, alpha=self.alpha, n_seeds=self.n_seeds).fit(X, y, sample_weight, aux)
        # Escalas fijadas con el set de entrenamiento (no con el lote que se
        # predice: eso usaria informacion de otros dias del mismo mes).
        self.sc_ = float(np.std(_logit(self.clf_.predict_proba(X)))) + 1e-12
        self.sr_ = float(np.std(_logit(self.reg_.predict_proba(X)))) + 1e-12
        return self

    def predict_components(self, X):
        return self.clf_.predict_components(X).join(self.reg_.predict_components(X))

    def predict_proba(self, X):
        c = self.clf_.predict_proba(X)
        r = self.reg_.predict_proba(X)
        # El regresor produce pseudo-probabilidades mas concentradas en 0.5:
        # se promedian los logits estandarizados para que ambos pesen igual.
        z = 0.5 * (_logit(c) / self.sc_ + _logit(r) / self.sr_)
        return 1 / (1 + np.exp(-z * self.sc_))


@dataclass
class ShuffledNull:
    """Control nulo: mismo modelo, pero con las etiquetas barajadas en cada
    reentrenamiento. Su ventaja fuera de muestra debe ser ~0; si no lo es, el
    pipeline esta filtrando informacion."""

    inner: object
    seed: int = 0

    def fit(self, X, y, sample_weight=None, aux=None):
        rng = np.random.default_rng(self.seed + len(y))
        perm = rng.permutation(len(y))
        y_sh = pd.Series(np.asarray(y)[perm], index=y.index)
        self.inner.fit(X, y_sh, sample_weight)
        return self

    def predict_components(self, X):
        return self.inner.predict_components(X)

    def predict_proba(self, X):
        return self.inner.predict_proba(X)


class PlattCalibrator:
    """Temperature scaling: p_cal = sigmoid(b * logit(p)).

    Sin intercepto a proposito. Un intercepto absorberia la frecuencia base
    del periodo de calibracion (el exceso es negativo algo mas seguido por el
    carry) y moveria el centro de 0.5, invirtiendo el signo de predicciones
    debiles. La pendiente b<1 encoge probabilidades sobreconfiadas hacia 0.5;
    b>1 las expande. El orden de las predicciones no cambia."""

    def fit(self, p: np.ndarray, y: np.ndarray):
        z = _logit(np.asarray(p)).reshape(-1, 1)
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000, fit_intercept=False).fit(z, y)
        self.temperature_ = 1.0 / float(self.lr_.coef_[0, 0]) if self.lr_.coef_[0, 0] != 0 else np.inf
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        z = _logit(np.asarray(p)).reshape(-1, 1)
        return self.lr_.predict_proba(z)[:, 1]


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def time_decay_weights(index: pd.DatetimeIndex, ref: pd.Timestamp, half_life_years: float | None) -> np.ndarray | None:
    """Peso exponencial por antiguedad: una observacion de hace ``half_life``
    anos pesa la mitad que una de hoy. None = todas pesan igual."""
    if half_life_years is None:
        return None
    age = (ref - index).days.values / 365.25
    w = 0.5 ** (age / half_life_years)
    return w / w.mean()


def walk_forward(
    df: pd.DataFrame,
    mask: pd.Series,
    features: list[str],
    start: str,
    end: str,
    model_factory,
    freq: str = "MS",
    half_life_years: float | None = None,
    train_start: str = config.TRAIN_START,
    embargo: int = config.EMBARGO_DAYS,
) -> pd.DataFrame:
    """Reentrena al inicio de cada periodo (``freq``: 'MS' mensual, 'YS'
    anual) con ventana expansiva y pesos por antiguedad.

    Para predecir el periodo [t0, t1) se entrena con [train_start, t0) menos
    un embargo de ``embargo`` jornadas. Devuelve probabilidades para todas las
    jornadas del periodo (incluidas las rancias: la estrategia igual decide).
    """
    preds = []
    idx = df.index
    pos = np.arange(len(idx))
    edges = pd.date_range(pd.Timestamp(start), pd.Timestamp(end) + pd.offsets.MonthBegin(1), freq=freq)
    for t0, t1 in zip(edges[:-1], edges[1:]):
        te = df[(idx >= t0) & (idx < t1) & (idx <= pd.Timestamp(end))]
        if te.empty:
            continue
        tr_end_pos = idx.searchsorted(t0) - embargo
        tr_idx = idx[(idx >= pd.Timestamp(train_start)) & (pos < tr_end_pos)]
        tr_idx = tr_idx[mask.reindex(tr_idx).fillna(False).values]
        tr = df.loc[tr_idx]
        w = time_decay_weights(tr.index, t0, half_life_years)
        m = model_factory()
        if isinstance(m, (RegEnsemble, Combo)):
            m.fit(tr[features], tr["y"], sample_weight=w, aux=tr)
        else:
            m.fit(tr[features], tr["y"], sample_weight=w)
        p = pd.DataFrame({"p_raw": m.predict_proba(te)}, index=te.index)
        if hasattr(m, "predict_components"):
            p = p.join(m.predict_components(te))
        p["fold"] = t0.year
        p["refit"] = t0
        preds.append(p)
    return pd.concat(preds)


def calibrate_walk_forward(pred: pd.DataFrame, y: pd.Series, mask: pd.Series, min_obs: int = 500,
                           col: str = "p_raw") -> pd.Series:
    """Calibra cada bloque de reentrenamiento con un Platt ajustado SOLO con
    predicciones fuera de muestra anteriores al bloque (y cuyo resultado ya se
    conocia). Mientras no haya ``min_obs`` predicciones previas se deja la
    probabilidad sin calibrar (periodo de calentamiento)."""
    out = pd.Series(np.nan, index=pred.index, name="p_cal")
    yy = y.reindex(pred.index)
    ok = mask.reindex(pred.index).fillna(False) & yy.notna()
    for t0 in sorted(pred["refit"].unique()):
        cur = pred["refit"] == t0
        past = ok & (pred.index < t0 - pd.Timedelta(days=1))
        if past.sum() < min_obs:
            out[cur] = pred.loc[cur, col]
            continue
        cal = PlattCalibrator().fit(pred.loc[past, col].values, yy[past].values)
        out[cur] = cal.transform(pred.loc[cur, col].values)
    return out
