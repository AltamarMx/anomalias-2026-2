import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 019b · Exploración de la serie `tdb`

    Esta libreta es la **parte práctica**. Asume que ya leíste
    **`019a_tdb_teoria.py`** y tienes en mente la fórmula

    $$
    \Phi_P(L^{s})\,\phi_p(L)\,(1-L^{s})^{D}\,(1-L)^{d}\,y_t
    \;=\;
    \Theta_Q(L^{s})\,\theta_q(L)\,\varepsilon_t
    $$

    Aquí vamos a mirar la serie `tdb` (temperatura de bulbo seco) del
    ClimaLab y a *traducir* lo que veamos a parámetros concretos
    `(p, d, q)(P, D, Q, s)`.

    **El recorrido es:**

    1. Cargar la serie y verla a 4 resoluciones (minuto / 10 min / hora / día).
    2. Perfil diario y anual — ¿hay estacionalidad anual además de la diaria?
    3. ACF / PACF — relacionar los picos con `p`, `q`, `P`, `Q`.
    4. Pruebas de estacionariedad (ADF, KPSS) — decidir `d` y `D`.
    5. Cierre: ¿qué órdenes intentar primero en la libreta 020?
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
    return acf, adfuller, go, kpss, make_subplots, mo, np, pacf, pd


@app.cell
def _(pd):
    F = "data/ClimaLab_2023-05-31_2025-06-20.parquet"
    tdb_full = pd.read_parquet(F, columns=["tdb"])["tdb"]
    tdb_full.index.name = "date"

    print(f"Observaciones: {len(tdb_full):,}")
    print(f"Rango: {tdb_full.index.min()} → {tdb_full.index.max()}")
    print(f"Frecuencia nativa ~1 min · NaNs: {tdb_full.isna().sum()}")
    print(f"Stats (°C): min={tdb_full.min():.2f}  media={tdb_full.mean():.2f}  max={tdb_full.max():.2f}")
    return (tdb_full,)


@app.cell
def _(mo, tdb_full):
    # Pre-computamos los 4 resamples — los usan tanto las gráficas como
    # las celdas de perfiles / ACF / estacionariedad de más abajo.
    tdb_min   = tdb_full
    tdb_10min = tdb_full.resample("10min").mean()
    tdb_h     = tdb_full.resample("h").mean().dropna()
    tdb_d     = tdb_full.resample("D").mean()

    mo.md(
        "**Resamples disponibles** — "
        f"1 min: `{len(tdb_min):,}` obs · "
        f"10 min: `{len(tdb_10min):,}` obs · "
        f"hora: `{len(tdb_h):,}` obs · "
        f"día: `{len(tdb_d):,}` obs.  "
        "El *range slider* debajo de cada gráfica sirve para acercarse a "
        "ventanas específicas."
    )
    return tdb_10min, tdb_d, tdb_h, tdb_min


@app.cell
def _(pd, tdb_10min, tdb_d, tdb_h, tdb_min):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # Cada panel muestra una *ventana* adaptada a su resolución, terminando todas
    # en la última fecha disponible. Así se aprecia la textura de cada resample
    # en vez de ver siempre la misma curva muestreada visualmente.
    _end = tdb_min.index.max()
    _panels = [
        (tdb_min.loc[_end - pd.Timedelta(days=1):],     "1 min", "última 24 h"),
        (tdb_10min.loc[_end - pd.Timedelta(days=7):],   "10 min", "última semana"),
        (tdb_h.loc[_end - pd.Timedelta(days=30):],      "Hora", "último mes"),
        (tdb_d,                                          "Día", "serie completa"),
    ]

    fig_all, axes = plt.subplots(4, 1, figsize=(11, 9))

    for _ax, (_s, _res, _win) in zip(axes, _panels):
        _ax.plot(_s.index, _s.values, linewidth=0.7, color="#2b6cb0")
        _ax.set_title(f"{_res} · {_win} · {len(_s):,} obs", loc="left", fontsize=10)
        _ax.set_ylabel("tdb (°C)")
        _ax.grid(True, alpha=0.3)
        _ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        _ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(_ax.xaxis.get_major_locator()))

    fig_all.suptitle("Temperatura de bulbo seco — la misma serie a 4 resoluciones", fontsize=12, y=0.995)
    fig_all.tight_layout()
    fig_all
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## B.2 Perfiles diario y anual

    Dos vistas que hacen visible la **estacionalidad**:

    ### Perfil diario — un boxplot por hora del día

    Para cada una de las 24 horas, dibujamos un boxplot con todas las
    temperaturas observadas a esa hora a lo largo de los ~2 años de datos.
    El eje X es la hora (0 → 23) y el eje Y son grados centígrados.

    **¿Qué quiere decir "las cajas se separan"?** Que el boxplot de cada hora
    queda en una *altura distinta* del eje Y, sin sobreponerse mucho con sus
    vecinos. Por ejemplo: las cajas de las **14–16 h** (calentamiento solar)
    flotan arriba, las de **04–07 h** (madrugada) se hunden abajo, y entre
    ambas hay un salto claro de varios grados.

    | Si ves esto en el boxplot | Significa |
    |---|---|
    | 24 cajas claramente escalonadas, alturas muy distintas | **ciclo diario fuerte** → vale la pena usar `s = 24`. |
    | Cajas a la misma altura, todas mezcladas alrededor del promedio | no hay ciclo diario → `s = 24` no aporta. |
    | Algunas horas con cajas mucho más altas/anchas (más varianza) | hay horas más "ruidosas" — útil para ajustar luego `p, q`. |

    ### Perfil anual — mapa de calor mes × hora-del-día

    Un *heatmap*: filas = mes (01-12), columnas = hora-del-día (0-23), color =
    temperatura promedio. Lee así:

    - Si las filas **cambian de color** entre invierno y verano → hay
      **estacionalidad anual** (los meses son sistemáticamente más fríos o más
      calientes).
    - Si la franja horaria caliente **se desplaza** con el mes (p. ej. arriba
      más temprano en verano que en invierno) → el ciclo diario también
      cambia de forma con el año, no solo de nivel.
    """)
    return


@app.cell
def _(go, tdb_h):
    hours = tdb_h.index.hour

    fig_daily = go.Figure()
    for h in range(24):
        _v = tdb_h[hours == h]
        fig_daily.add_trace(go.Box(
            y=_v, name=f"{h:02d}",
            boxpoints=False,
            marker_color="#2b6cb0",
            line=dict(width=1),
            showlegend=False,
        ))

    fig_daily.update_layout(
        title="Perfil diario de tdb — distribución por hora del día",
        xaxis_title="hora del día", yaxis_title="tdb (°C)",
        height=380, margin=dict(l=40, r=20, t=50, b=40),
        template="plotly_white",
    )
    fig_daily
    return


@app.cell
def _(go, pd, tdb_full):
    # Mapa de calor mes-del-año × hora-del-día sobre TODA la serie disponible
    # (no la ventana), para que la estacionalidad anual sea visible.
    ts_year = tdb_full.resample("h").mean().dropna()
    pivot = (
        pd.DataFrame({"tdb": ts_year, "hour": ts_year.index.hour, "month": ts_year.index.month})
        .groupby(["month", "hour"])["tdb"].mean()
        .unstack("hour")
    )

    fig_year = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,                              # hora del día
        y=[f"{m:02d}" for m in pivot.index],          # mes
        colorscale="RdBu_r",
        colorbar=dict(title="°C"),
        hovertemplate="mes %{y} · hora %{x:02d} · %{z:.2f} °C<extra></extra>",
    ))
    fig_year.update_layout(
        title="Perfil anual de tdb — temperatura promedio por mes × hora del día",
        xaxis_title="hora del día", yaxis_title="mes",
        height=380, margin=dict(l=40, r=20, t=50, b=40),
        template="plotly_white",
    )
    fig_year
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## B.3 ACF / PACF — ¿dónde están los picos?

    Estas dos funciones miden la **correlación** entre la serie y versiones
    *rezagadas* de sí misma. Son la mejor pista para elegir órdenes.

    - **ACF (autocorrelación)** — picos en los rezagos `1, 2, …` sugieren MA
      (`q`). Picos en los rezagos `s, 2s, …` sugieren MA estacional (`Q`).
    - **PACF (autocorrelación parcial)** — picos en los rezagos `1, 2, …`
      sugieren AR (`p`). Picos en los rezagos `s, 2s, …` sugieren AR
      estacional (`P`).

    **Importante:** ACF/PACF aquí están calculadas sobre la serie *horaria*
    (`tdb_h`), independientemente de la frecuencia del selector — para que
    `lag = 24` siempre signifique "un día atrás".
    """)
    return


@app.cell
def _(acf, go, make_subplots, np, pacf, tdb_h):
    nlags = 500

    n_obs = len(tdb_h)
    ci_band = 1.96 / np.sqrt(n_obs)

    acf_vals = acf(tdb_h.values, nlags=nlags, fft=True)
    pacf_vals = pacf(tdb_h.values, nlags=nlags, method="ywm")

    fig_acf = make_subplots(rows=1, cols=2, subplot_titles=("ACF (horaria)", "PACF (horaria)"))

    for _col, (_v, _name) in enumerate([(acf_vals, "ACF"), (pacf_vals, "PACF")], start=1):
        fig_acf.add_trace(
            go.Bar(x=list(range(len(_v))), y=_v,
                   name=_name, marker_color="#2b6cb0",
                   hovertemplate="lag=%{x}<br>%{y:.3f}<extra></extra>"),
            row=1, col=_col,
        )
        fig_acf.add_hline(y=ci_band, line=dict(dash="dash", color="grey"), row=1, col=_col)
        fig_acf.add_hline(y=-ci_band, line=dict(dash="dash", color="grey"), row=1, col=_col)
        for _k in range(24, nlags + 1, 24):
            fig_acf.add_vline(x=_k, line=dict(color="orange", width=1, dash="dot"),
                              row=1, col=_col)

    fig_acf.update_layout(
        height=360, showlegend=False,
        margin=dict(l=40, r=20, t=50, b=40),
        template="plotly_white",
    )
    fig_acf.update_xaxes(title_text="lag (horas)")
    fig_acf
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## B.4 ¿Estacionaria? Pruebas ADF y KPSS

    SARIMA exige que la serie sea **estacionaria** *después* de aplicar las
    diferencias `d` y `D`. Probamos cuatro transformaciones y, en cada una,
    corremos dos pruebas con hipótesis nulas **opuestas**:

    | Prueba | $H_0$ | Conclusión cuando **rechazamos** $H_0$ |
    |---|---|---|
    | **ADF** (Dickey-Fuller aumentado) | "la serie tiene raíz unitaria" | la serie es estacionaria ✓ |
    | **KPSS** (Kwiatkowski et al.) | "la serie es estacionaria" | la serie **no** es estacionaria ✗ |

    Como sus $H_0$ están invertidas, **ADF bajo + KPSS alto = estacionaria con
    alta confianza**. Las cuatro transformaciones son:

    1. `y` — sin diferenciar.
    2. `Δy = (1 − L) y` — diferencia regular (`d = 1`).
    3. `Δ₂₄ y = (1 − L²⁴) y` — diferencia estacional (`D = 1, s = 24`).
    4. `Δ Δ₂₄ y = (1 − L)(1 − L²⁴) y` — ambas.

    Buscamos la **combinación de menor `d + D`** que ya nos dé estacionariedad
    en *ambas* pruebas.
    """)
    return


@app.cell
def _(adfuller, kpss, pd, tdb_h):
    def _stationarity_row(name, series, *, d, D):
        s = series.dropna()
        if len(s) > 5000:
            s = s.iloc[:: max(1, len(s) // 5000)]
        adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
        kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
        return {
            "d": d,
            "D": D,
            "transformación": name,
            "n": len(s),
            "ADF stat": round(adf_stat, 3),
            "ADF p": round(adf_p, 4),
            "ADF": "estacionaria" if adf_p < 0.05 else "NO estacionaria",
            "KPSS stat": round(kpss_stat, 3),
            "KPSS p": round(kpss_p, 4),
            "KPSS": "estacionaria" if kpss_p > 0.05 else "NO estacionaria",
        }

    _rows = [
        _stationarity_row("y",        tdb_h,                  d=0, D=0),
        _stationarity_row("Δy",       tdb_h.diff(),           d=1, D=0),
        _stationarity_row("Δ₂₄ y",    tdb_h.diff(24),         d=0, D=1),
        _stationarity_row("Δ Δ₂₄ y",  tdb_h.diff().diff(24),  d=1, D=1),
    ]
    stat_table = pd.DataFrame(_rows)
    stat_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## B.5 Cierre — ¿qué `(p, d, q)(P, D, Q, s)` intentar primero?

    Junta lo que viste arriba con lo que aprendiste en la Parte A:

    | Evidencia en los datos | Decisión sobre los órdenes |
    |---|---|
    | El plot temporal muestra un **ciclo claro de ~24 horas**. | `s = 24`. |
    | El perfil diario tiene **cajas separadas** por hora. | Hay estacionalidad diaria → vale la pena `D ≥ 1`. |
    | El perfil anual muestra **deriva mes a mes**. | Sí hay estacionalidad anual — pero modelarla con `s = 8760` es caro; la dejamos para más adelante. |
    | ACF cae lento y tiene picos enormes en **lag 24, 48, 72…** | Componente estacional dominante → `D = 1`. |
    | ADF y KPSS **rechazan estacionariedad** en la serie cruda. | Hace falta al menos una diferencia. |
    | **Sola `Δ₂₄`** (`D=1, s=24`) ya deja la serie estacionaria. | `d = 0, D = 1` es suficiente — no diferenciamos de más. |
    | Quedan picos en lags pequeños tras diferenciar. | Reservamos `p, q ∈ {1, 2}` para la próxima libreta. |

    ### Receta de arranque para la libreta 020

    > **`SARIMA(1, 0, 1)(1, 1, 1)₂₄`** sobre una ventana hourly de 3–4 semanas.

    Es el punto medio: `d = 0`, `D = 1`, con AR y MA simples a ambas escalas.
    Desde ahí, mirando la ACF de los **residuos**, iteras subiendo/bajando
    perillas como dijimos en la sección A.5.

    ---

    ### Lo que sigue

    - **020** — fitear ese modelo, ver diagnósticos, iterar.
    - **021** — convertirlo en un pronosticador y compararlo con baselines.
    - **022** — usarlo como detector de anomalías sobre la serie completa.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
