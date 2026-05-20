import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 020 · Configurar SARIMA sobre `tdb`

    En **019a** vimos la fórmula y los siete parámetros. En **019b** confirmamos
    con los datos que:

    - el ciclo diario es fuerte → `s = 24`,
    - la diferenciación estacional `D = 1` ya deja la serie estacionaria → `d = 0`,
    - quedan picos en lags pequeños tras diferenciar → reservar `p, q ∈ {1, 2}`.

    **En esta libreta vamos a:**

    1. Cortar una ventana de entrenamiento sobre `tdb` horaria y separar un
       holdout.
    2. Ajustar SARIMA con un **widget interactivo** para `(p, d, q)(P, D, Q, s)`.
    3. Leer **diagnósticos**: AIC/BIC en entrenamiento, RMSE/MAE en holdout,
       residuos, ACF/PACF de residuos y Ljung-Box.
    4. Iterar las perillas usando una pequeña **receta**: "si ves esto →
       mueve esto".

    > Recordatorio compacto de la fórmula:
    >
    > $$
    > \Phi_P(L^{s})\,\phi_p(L)\,(1-L^{s})^{D}\,(1-L)^{d}\,y_t
    > \;=\;
    > \Theta_Q(L^{s})\,\theta_q(L)\,\varepsilon_t
    > $$
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import acf, pacf
    from statsmodels.stats.diagnostic import acorr_ljungbox

    import warnings
    warnings.filterwarnings("ignore")
    return SARIMAX, acorr_ljungbox, mdates, mo, np, pd, plt


@app.cell
def _(mo, pd):
    F = "data/ClimaLab_2023-05-31_2025-06-20.parquet"
    _tdb_raw = pd.read_parquet(F, columns=["tdb"])["tdb"]
    _tdb_raw.index.name = "date"

    # Trabajamos a frecuencia horaria — manejable y `s = 24` se mapea al ciclo diario.
    tdb_h = _tdb_raw.resample("h").mean().dropna()
    tdb_h.name = "tdb"

    mo.md(
        f"Serie horaria: **{len(tdb_h):,}** obs · "
        f"{tdb_h.index.min().date()} → {tdb_h.index.max().date()}."
    )
    return (tdb_h,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ventana de entrenamiento y holdout

    SARIMA es lento si le das demasiados datos. Cortamos una ventana acotada de
    **entrenamiento** y reservamos las últimas **`h`** horas como **holdout**
    (ahí vamos a medir el pronóstico fuera de muestra).

    - **Inicio**: fecha desde donde empieza el entrenamiento.
    - **Semanas de entrenamiento**: cuánto entrenamiento usar (3–8 semanas
      suelen ser un buen balance: suficiente para que el ciclo diario se
      estime, no tanto como para que el ajuste se eternice).
    - **Horizonte h**: cuántas horas del final reservamos para holdout.
    """)
    return


@app.cell
def _(mo, pd, tdb_h):
    start_picker = mo.ui.date(
        start=tdb_h.index.min().date(),
        stop=tdb_h.index.max().date(),
        value=pd.Timestamp("2025-04-01").date(),
        label="**Inicio de la ventana**",
    )
    weeks_slider = mo.ui.slider(
        1, 12, value=4, step=1,
        label="**Semanas de entrenamiento**",
        show_value=True,
    )
    horizon_number = mo.ui.number(
        start=12, stop=14 * 24, value=48, step=12,
        label="**Horizonte holdout (horas)**",
    )
    mo.hstack([start_picker, weeks_slider, horizon_number], gap=2)
    return horizon_number, start_picker, weeks_slider


@app.cell
def _(horizon_number, mo, pd, start_picker, tdb_h, weeks_slider):
    _t0 = pd.Timestamp(start_picker.value)
    _t1 = _t0 + pd.Timedelta(weeks=int(weeks_slider.value))
    _h  = int(horizon_number.value)

    # .asfreq("h") fija la frecuencia del índice (necesario para que SARIMAX
    # devuelva el forecast con DatetimeIndex). Si hay horas faltantes, asfreq
    # las rellena con NaN; los interpolamos linealmente para no contaminar el
    # fit ni la ACF/PACF.
    _window = tdb_h.loc[_t0:_t1].asfreq("h").interpolate("time")
    y_train = _window.iloc[:-_h]
    y_test  = _window.iloc[-_h:]

    mo.md(
        f"**Train:** {len(y_train):,} obs · "
        f"{y_train.index[0]} → {y_train.index[-1]}  \n"
        f"**Test:**  {len(y_test):,} obs · "
        f"{y_test.index[0]} → {y_test.index[-1]}  \n"
        f"freq = `{y_train.index.freqstr}` · NaN tras interpolación: "
        f"`{int(_window.isna().sum())}`"
    )
    return y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Widget de órdenes `(p, d, q)(P, D, Q, s)`

    Mueve las perillas y pulsa **Fit SARIMA** para reajustar. Los valores por
    defecto vienen de la receta sugerida al final de **019b**:
    `SARIMA(1, 0, 1)(1, 1, 1)₂₄`.

    | Parte | Significa |
    |---|---|
    | **(p, d, q)** | dinámica de corto plazo + diferencias regulares. |
    | **(P, D, Q, s)** | dinámica en saltos de `s` + diferencias estacionales. |
    """)
    return


@app.cell(hide_code=True)
def _(SARIMAX, mo, np, order_form, y_test, y_train):
    cfg = order_form.value
    mo.stop(cfg is None, mo.md("👆 Elige órdenes y pulsa **Fit SARIMA**."))

    order_         = (cfg["p"], cfg["d"], cfg["q"])
    seasonal_order = (cfg["P"], cfg["D"], cfg["Q"], cfg["s"])

    with mo.status.spinner("Ajustando SARIMA…"):
        res = SARIMAX(
            y_train,
            order=order_,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

    _fc    = res.get_forecast(steps=len(y_test))
    y_hat  = _fc.predicted_mean
    ci_fc  = _fc.conf_int(alpha=0.05)

    _rmse = float(np.sqrt(((y_test - y_hat) ** 2).mean()))
    _mae  = float((y_test - y_hat).abs().mean())

    mo.md(
        f"**SARIMA{order_} × {seasonal_order}** &nbsp;·&nbsp; "
        f"AIC = `{res.aic:.2f}` &nbsp;·&nbsp; "
        f"BIC = `{res.bic:.2f}` &nbsp;·&nbsp; "
        f"RMSE(test) = `{_rmse:.2f} °C` &nbsp;·&nbsp; "
        f"MAE(test) = `{_mae:.2f} °C`"
    )
    return ci_fc, res, seasonal_order, y_hat


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Panel de diagnóstico

    Cuatro vistas que sirven para iterar las perillas:

    | Panel | Qué mirar |
    |---|---|
    | **Forecast vs holdout** | La línea roja (pronóstico) debe seguir el patrón diario de la negra (verdad). La banda rosa es el intervalo 95%. |
    | **Residuos** | Deben oscilar alrededor de 0 sin patrones visibles (sin tendencia, sin ciclos). |
    | **ACF de residuos** | Casi todas las barras dentro de la banda azul. Picos en lag 24, 48 → falta `D = 1` o `Q = 1`. |
    | **PACF de residuos** | Picos persistentes en lag 24 → necesitas más `P`. |

    > Antes de leer la ACF/PACF descartamos las primeras observaciones
    > (*burn-in*): el filtro de Kalman necesita unos pasos para entrar en
    > régimen y los residuos iniciales son artefacto.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    order_form = (
        mo.md(
            """
            **No estacional:** p = {p}, d = {d}, q = {q}

            **Estacional:**    P = {P}, D = {D}, Q = {Q}, s = {s}
            """
        )
        .batch(
            p=mo.ui.slider(0, 5, value=1, label="p"),
            d=mo.ui.slider(0, 2, value=0, label="d"),
            q=mo.ui.slider(0, 5, value=1, label="q"),
            P=mo.ui.slider(0, 2, value=1, label="P"),
            D=mo.ui.slider(0, 1, value=1, label="D"),
            Q=mo.ui.slider(0, 2, value=1, label="Q"),
            s=mo.ui.number(2, 168, value=24, label="s"),
        )
        .form(submit_button_label="Fit SARIMA")
    )
    order_form
    return (order_form,)


@app.cell(hide_code=True)
def _(ci_fc, mdates, plt, res, seasonal_order, y_hat, y_test, y_train):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    # Quitar el burn-in (al menos s pasos, mínimo 24)
    _burn = max(seasonal_order[3], 24)
    resid = res.resid.iloc[_burn:]

    fig_diag, _axes = plt.subplots(2, 2, figsize=(12, 7))

    # (0,0) Forecast vs holdout — pintamos con matplotlib directo para tener
    # control del formato de fechas (pandas .plot() con DatetimeIndex usa su
    # propio formateador y no lo cede bien a mdates).
    _ax = _axes[0, 0]
    _train_tail = y_train.iloc[-7 * 24:]
    _ax.plot(_train_tail.index, _train_tail.values,
             label="train (últimos 7 d)", color="#888", linewidth=0.8)
    _ax.plot(y_test.index, y_test.values,
             label="test", color="black", linewidth=1.2)
    _ax.plot(y_hat.index, y_hat.values,
             label="forecast", color="red", linewidth=1.2)
    _ax.fill_between(ci_fc.index, ci_fc.iloc[:, 0], ci_fc.iloc[:, 1],
                     color="red", alpha=0.15, label="IC 95%")
    _ax.legend(loc="upper left", fontsize=8)
    _ax.set_title("Forecast vs holdout")
    _ax.set_ylabel("tdb (°C)")
    _ax.grid(True, alpha=0.3)
    _loc0 = mdates.AutoDateLocator(maxticks=6)
    _ax.xaxis.set_major_locator(_loc0)
    _ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(_loc0))

    # (0,1) Residuos
    _ax = _axes[0, 1]
    _ax.plot(resid.index, resid.values, color="#2b6cb0", linewidth=0.7)
    _ax.axhline(0, color="k", lw=0.5)
    _ax.set_title(f"Residuos (post burn-in, n={len(resid):,})")
    _ax.grid(True, alpha=0.3)
    _loc1 = mdates.AutoDateLocator(maxticks=6)
    _ax.xaxis.set_major_locator(_loc1)
    _ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(_loc1))

    # (1,0) ACF de residuos
    plot_acf(resid, ax=_axes[1, 0], lags=48)
    _axes[1, 0].set_title("ACF de residuos")
    _axes[1, 0].set_xlabel("lag (horas)")

    # (1,1) PACF de residuos
    plot_pacf(resid, ax=_axes[1, 1], lags=48, method="ywm")
    _axes[1, 1].set_title("PACF de residuos")
    _axes[1, 1].set_xlabel("lag (horas)")

    fig_diag.tight_layout()
    fig_diag
    return (resid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ljung-Box — "¿quedó algo de estructura en los residuos?"

    Es un test estadístico que examina, en bloque, varios rezagos de la ACF de
    los residuos a la vez. Su hipótesis nula es:

    > $H_0$: las primeras `k` autocorrelaciones son cero — los residuos son
    > ruido blanco.

    Si **el p-value es alto** (típicamente > 0.05), **no podemos rechazar** que
    los residuos sean ruido blanco → el modelo ya "consumió" la estructura.
    Si es bajo, sigue habiendo señal sin explicar.

    Probamos en rezagos significativos para `tdb`: 12, 24 (estacional), 48.
    """)
    return


@app.cell(hide_code=True)
def _(acorr_ljungbox, np, resid):
    _lb_lags = [12, 24, 48]
    lb = acorr_ljungbox(resid, lags=_lb_lags, return_df=True)
    lb = lb.assign(**{
        "veredicto": np.where(lb["lb_pvalue"] > 0.05,
                              "ruido blanco ✓",
                              "estructura residual ✗"),
    })
    lb.index.name = "lag"
    lb.round(4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recetario — ¿qué perilla mover?

    Mira la **ACF/PACF de residuos** y el **Ljung-Box** después de cada fit.
    Una iteración por vez. Sube de a `1` cada parámetro y vuelve a fitear.

    | Síntoma en los residuos | Mueve esto | Por qué |
    |---|---|---|
    | Pico grande en lag **24** en la ACF | `Q = Q + 1` o `D = 1` | falta absorber el "error de ayer a la misma hora". |
    | Decaimiento lento en lags 24, 48, 72 | `P = P + 1` (rara vez > 2) | el modelo necesita memoria estacional autorregresiva. |
    | Picos en lags pequeños (1–5) en la ACF | `q = q + 1` | dinámica de corto plazo en los errores. |
    | Picos en lags pequeños en la PACF | `p = p + 1` | dinámica de corto plazo en los valores. |
    | Residuos con tendencia visible | `d = 1` | la serie no estaba realmente sin tendencia. |
    | Residuos con varianza creciente | considerar log(`tdb`) o BoxCox | heterocedasticidad — fuera del alcance de SARIMA puro. |
    | AIC baja al añadir parámetro **y** RMSE en holdout baja | mantén el cambio | el modelo mejora. |
    | AIC baja pero RMSE no | descarta el cambio | sobreajuste. |

    ### Reglas de oro

    1. **`d + D ≤ 2`** — diferenciar de más amplifica ruido.
    2. **Empieza simple.** Una iteración por perilla; no muevas tres a la vez.
    3. **La verdad fuera de muestra manda.** AIC/BIC son referencias; RMSE en
       holdout es el voto definitivo.
    """)
    return


if __name__ == "__main__":
    app.run()
