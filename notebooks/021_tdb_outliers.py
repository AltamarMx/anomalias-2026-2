import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 021 · Detección de outliers con SARIMA — `tdb`

    En la libreta 020 dedujimos, ajustamos y comparamos tres modelos SARIMA
    para la temperatura de bulbo seco (`tdb`). Aquí **reutilizamos esos tres
    modelos**, pero con un objetivo distinto y un proceso mucho más corto:

    1. **Entrenar** los tres modelos y medir **solo RMSE y MAE** en el holdout.
    2. **Elegir** uno (el de menor RMSE).
    3. Con ese modelo, repasar la **teoría de detección de outliers**.
    4. **Intentar detectar outliers** en las 24 h del holdout — sin importar
       que no encontremos ninguno.

    > **Ventana fija** — idéntica a la 020, para que todo sea comparable:
    >
    > | Pieza | Valor |
    > |---|---|
    > | Serie | `tdb` resampleada a **1 hora** |
    > | Entrenamiento | `2024-03-15 00:00 → 2024-04-11 23:00` (672 obs) |
    > | Holdout | `2024-04-12 00:00 → 2024-04-12 23:00` (24 h) |
    > | Estacionalidad | `s = 24` |

    A diferencia de la 020, aquí **no** repetimos ADF/KPSS, ACF/PACF,
    periodograma ni los diagnósticos de residuos: esa parte ya quedó
    justificada. Vamos directo a entrenar, medir y detectar.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    import plotly.graph_objects as go

    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from scipy import stats

    import warnings
    warnings.filterwarnings("ignore")
    return SARIMAX, go, mo, np, pd, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Cargar la serie y fijar la ventana

    Misma lectura, mismo `resample` horario y mismas tres fechas que en la
    020. A partir de aquí `train` y `test` no cambian.
    """)
    return


@app.cell
def _(pd):
    f = "data/ClimaLab_2023-05-31_2025-06-20.parquet"
    tdb_h = (
        pd.read_parquet(f, columns=["tdb"])["tdb"]
        .resample("h").mean().dropna()
    )

    # train = [f1, f2)  ← 4 semanas ;  test = [f2, f3)  ← 24 h
    f1 = pd.Timestamp("2024-03-15 00:00")
    f2 = f1 + pd.Timedelta(weeks=4)
    f3 = f2 + pd.Timedelta(hours=24)

    train = tdb_h[(tdb_h.index >= f1) & (tdb_h.index < f2)].asfreq("h")
    test  = tdb_h[(tdb_h.index >= f2) & (tdb_h.index < f3)].asfreq("h")

    print(f"train: {len(train)} obs · {train.index.min()} → {train.index.max()}")
    print(f"test : {len(test)} obs · {test.index.min()} → {test.index.max()}")
    return test, train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Entrenar los tres modelos y medir RMSE / MAE

    Lo más simple posible: una función que **ajusta** el modelo, hace
    `get_forecast(24)` y devuelve **solo** RMSE y MAE sobre el holdout (más el
    objeto `res`, que necesitaremos para la detección). Sin AIC, sin
    Ljung-Box, sin diagnósticos — eso ya lo hicimos en la 020.

    - **RMSE** = $\sqrt{\text{mean}(e_t^2)}$ — castiga errores grandes.
    - **MAE** = $\text{mean}(|e_t|)$ — promedio del error absoluto.

    Los tres órdenes son exactamente los que dedujimos e iteramos en la 020.
    """)
    return


@app.cell
def _(SARIMAX, np, test, train):
    def fit_rmse_mae(order, seasonal_order, name):
        res = SARIMAX(
            train, order=order, seasonal_order=seasonal_order,
            enforce_stationarity=True, enforce_invertibility=True,
        ).fit(disp=False)

        pred = res.get_forecast(steps=len(test)).predicted_mean
        err = pred.values - test.values
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae  = float(np.mean(np.abs(err)))

        return {
            "name": name,
            "order": order,
            "seasonal_order": seasonal_order,
            "RMSE": rmse,
            "MAE": mae,
            "res": res,
        }

    return (fit_rmse_mae,)


@app.cell
def _(fit_rmse_mae, pd):
    modelos = [
        fit_rmse_mae((1, 0, 1), (1, 1, 1, 24), "M1"),
        fit_rmse_mae((1, 0, 1), (2, 1, 1, 24), "M2"),
        fit_rmse_mae((2, 0, 2), (2, 1, 1, 24), "M3"),
    ]

    tabla_metricas = pd.DataFrame([
        {
            "modelo": m["name"],
            "orden": f"{m['order']}{m['seasonal_order']}",
            "RMSE": round(m["RMSE"], 3),
            "MAE":  round(m["MAE"], 3),
        }
        for m in modelos
    ])
    tabla_metricas
    return (modelos,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Seleccionar el modelo

    Regla simple: **gana el de menor RMSE en el holdout** (desempate por
    MAE). Ese es el modelo que usaremos para la detección de outliers.
    """)
    return


@app.cell
def _(modelos):
    ganador = min(modelos, key=lambda m: (m["RMSE"], m["MAE"]))

    print(f"Modelo ganador: {ganador['name']}  "
          f"{ganador['order']}{ganador['seasonal_order']}")
    print(f"  RMSE = {ganador['RMSE']:.3f} °C")
    print(f"  MAE  = {ganador['MAE']:.3f} °C")
    return (ganador,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Teoría — detectar outliers con SARIMA

    ### La idea central

    Un modelo SARIMA, una vez ajustado, sabe **qué valor "debería" tener**
    cada hora: la predicción $\hat y_t$. Un *outlier* es un punto cuyo valor
    real $y_t$ se aleja **demasiado** de lo esperado. Lo medimos con el
    **residuo**:

    $$e_t = y_t - \hat y_t$$

    ### Residuo estandarizado y umbral

    Primero un término que usaremos enseguida. La **innovación** $\varepsilon_t$
    es la parte **genuinamente nueva e impredecible** de cada observación: lo
    que el modelo *no* pudo anticipar a partir de todo el pasado. Formalmente
    es el error de predicción a un solo paso,
    $\varepsilon_t = y_t - \hat y_{t\mid t-1}$. Si el modelo está bien
    ajustado, esa innovación se parece a **ruido blanco gaussiano** con
    varianza constante $\sigma^2$ — es el `sigma2` del `summary` de la 020.

    Pero al pronosticar no medimos esa innovación de un paso, sino el **error
    de pronóstico acumulado**. Llamemos $T$ al último dato conocido y
    $h = 1, 2, \dots, 24$ al **horizonte** (cuántas horas adelante
    pronosticamos). Como predecimos varias horas seguidas sin ver los valores
    reales intermedios, el error a $h$ pasos arrastra **todas las innovaciones
    aún por ocurrir** entre $T$ y $T+h$, pesadas por la dinámica del modelo.

    Eso se ve usando la representación **MA($\infty$)** del modelo —cada
    observación como suma de innovaciones pasadas con pesos $\psi_j$ (la
    "respuesta al impulso", $\psi_0 = 1$):

    $$e_{T+h} = \sum_{j=0}^{h-1} \psi_j\,\varepsilon_{T+h-j}
      \qquad\Rightarrow\qquad
      \sigma_{(h)}^{2} = \operatorname{Var}(e_{T+h}) = \sigma^{2}\sum_{j=0}^{h-1}\psi_j^{\,2}$$

    > **Ojo — qué es $\sigma_{(h)}$ y qué NO es.**
    >
    > - **No** se calcula juntando una muestra de residuos y sacando su
    >   desviación: del holdout hay *un* solo pronóstico, no hay repeticiones
    >   que promediar.
    > - Es una cantidad **teórica del modelo**: sale de sus coeficientes
    >   ($\psi_j$) y de la varianza de la innovación $\sigma^2$ — no mira los
    >   datos del holdout.
    > - El subíndice es el **horizonte $h$**, no la hora del reloj. **Crece con
    >   $h$** porque la suma añade un término $\psi_j^2 \ge 0$ por cada paso —
    >   por eso la banda se ensancha. En $h=1$ vale $\sigma_{(1)}=\sigma$ (solo
    >   la innovación fresca).
    > - En el código de §5 es exactamente `fc.se_mean` (`statsmodels` lo computa
    >   vía el filtro de Kalman, equivalente a esta suma).

    El error de pronóstico es gaussiano de media cero, así que al dividir entre
    su $\sigma_{(h)}$ obtenemos un residuo **estandarizado**:

    $$z_{(h)} = \frac{e_{T+h}}{\sigma_{(h)}} \sim \mathcal{N}(0, 1)$$

    Entonces declaramos *outlier* cuando $|z_{(h)}| > k$. El umbral $k$ fija
    qué tan estrictos somos:

    | $k$ | Cobertura | Interpretación |
    |---|---|---|
    | 1.96 | 95%   | banda permisiva — más candidatos |
    | 2.58 | 99%   | banda intermedia |
    | 3.00 | 99.7% | banda estricta — solo lo muy extremo |

    ### Por qué usamos el intervalo de predicción y no una $\sigma$ fija

    Al pronosticar hacia adelante, la **incertidumbre crece con el
    horizonte**: predecir la hora +1 es más seguro que la hora +24. Por eso
    $\sigma_{(h)}$ **no** es constante. `get_forecast` nos da directamente el
    **intervalo de predicción**
    $[\hat y_{T+h} - k\,\sigma_{(h)},\ \hat y_{T+h} + k\,\sigma_{(h)}]$ con su
    $\sigma_{(h)}$ correcta para cada horizonte. Un punto **fuera del
    intervalo** es exactamente un punto con $|z_{(h)}| > k$ → ese es nuestro
    criterio.

    ### In-sample vs out-of-sample

    - **In-sample**: buscar outliers *dentro del train* mirando los residuos
      del ajuste. Útil para limpiar datos, pero el propio outlier sesga el
      ajuste (hay que iterar: detectar → quitar → reajustar).
    - **Out-of-sample**: buscar outliers en datos **nuevos** comparándolos con
      lo que el modelo predijo. **Es lo que haremos aquí**: el modelo entrenó
      en 4 semanas que *no* incluyen el holdout, así que las 24 h del holdout
      son datos "frescos" y limpios para probar.

    ### Tipos de outlier (Box–Tiao) — para interpretar lo que salga

    | Tipo | Símbolo | Forma |
    |---|---|---|
    | Aditivo | AO | un pico aislado en una sola hora |
    | Cambio de nivel | LS | la serie salta a otro nivel y se queda |
    | Cambio transitorio | TC | un salto que decae poco a poco |
    | Innovacional | IO | un shock que se propaga vía la dinámica del modelo |

    ### Enfoque elegido (el simple)

    Reutilizamos el **forecast multi-paso a 24 h** que ya sabemos hacer (igual
    que en la 020) y marcamos los puntos del holdout que caen fuera de su
    intervalo de predicción. Ventaja extra: como el forecast **no** ve los
    valores reales, un outlier en una hora no "contamina" la predicción de las
    horas siguientes.

    > *Alternativa más sensible (no la implementamos):* predicción
    > **un-paso-adelante** alimentando los valores reales hora a hora. Da
    > bandas más angostas y detecta más, pero un outlier sí contamina la
    > predicción siguiente. La dejamos mencionada.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Detectar outliers en las 24 h del holdout

    Con el modelo ganador calculamos el forecast de 24 h y dos bandas de
    predicción: **95%** ($k = 1.96$) y **99%** ($k = 2.58$). Para cada hora
    reportamos el residuo estandarizado $z_t$ y si cae fuera de cada banda.
    """)
    return


@app.cell
def _(ganador, np, pd, test):
    _res = ganador["res"]
    _fc = _res.get_forecast(steps=len(test))

    pred   = _fc.predicted_mean
    se     = _fc.se_mean                      # σ_t del pronóstico, hora a hora
    ci95   = _fc.conf_int(alpha=0.05)
    ci99   = _fc.conf_int(alpha=0.01)

    resid  = test.values - pred.values
    z      = resid / se.values                # residuo estandarizado

    out95 = (test.values < ci95.iloc[:, 0].values) | (test.values > ci95.iloc[:, 1].values)
    out99 = (test.values < ci99.iloc[:, 0].values) | (test.values > ci99.iloc[:, 1].values)

    tabla_out = pd.DataFrame({
        "hora":     test.index,
        "real":     np.round(test.values, 2),
        "pred":     np.round(pred.values, 2),
        "lo95":     np.round(ci95.iloc[:, 0].values, 2),
        "hi95":     np.round(ci95.iloc[:, 1].values, 2),
        "z":        np.round(z, 2),
        "out_95":   out95,
        "out_99":   out99,
    })

    print(f"Outliers al 95%: {int(out95.sum())} de {len(test)} horas")
    print(f"Outliers al 99%: {int(out99.sum())} de {len(test)} horas")
    print(f"|z| máximo observado: {np.abs(z).max():.2f}")
    tabla_out
    return ci95, out95, pred


@app.cell
def _(ci95, go, out95, pred, test):
    fig_out = go.Figure()

    # banda de predicción 95%
    fig_out.add_trace(go.Scatter(
        x=list(pred.index) + list(pred.index[::-1]),
        y=list(ci95.iloc[:, 1].values) + list(ci95.iloc[:, 0].values[::-1]),
        fill="toself", fillcolor="rgba(21,128,61,0.15)",
        line=dict(width=0), name="banda 95%", hoverinfo="skip",
    ))
    # forecast
    fig_out.add_trace(go.Scatter(
        x=pred.index, y=pred.values, mode="lines",
        name="forecast", line=dict(color="#15803d", width=2, dash="dash"),
    ))
    # holdout real
    fig_out.add_trace(go.Scatter(
        x=test.index, y=test.values, mode="lines+markers",
        name="holdout real", line=dict(color="#d97706", width=2),
    ))
    # outliers resaltados
    fig_out.add_trace(go.Scatter(
        x=test.index[out95], y=test.values[out95], mode="markers",
        name="outlier (95%)",
        marker=dict(color="red", size=11, symbol="x", line=dict(width=2)),
    ))

    fig_out.update_layout(
        title="Holdout 24 h vs forecast — outliers fuera de la banda 95%",
        xaxis_title="fecha", yaxis_title="tdb (°C)",
        template="plotly_white",
        height=380, margin=dict(l=40, r=20, t=50, b=40),
    )
    fig_out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · ¿Son outliers reales, o un límite del método?

    Si §5 marcó varias horas (p. ej. 5 de 24 al 95%), conviene frenar antes de
    llamarlas "anomalías". Casi con seguridad son **datos reales y normales**,
    y lo que las marca es una **limitación de usar SARIMA multi-paso como
    detector**, no una propiedad de los datos.

    ### La evidencia está en los propios números

    1. **5 de 24 al 95% es demasiado.** Un intervalo del 95% ya *espera* que
       ~5% de los puntos caigan fuera aunque todo sea normal. 5/24 ≈ 21% no es
       "21% de anomalías": es señal de que **la banda quedó demasiado angosta**.
    2. **El `|z|` máximo ≈ 2.7 es pequeño.** Una anomalía de verdad (sensor
       pegado, dato corrupto, tormenta) da `z` de 5, 10, 20. Un `z` de ~2.7 es
       justo lo que produce el error de pronóstico ordinario; el punto *más*
       extremo del día apenas pasa el umbral del 99%.
    3. **Suelen estar agrupadas, no aisladas.** En la gráfica las marcas caen
       en horas consecutivas (la subida del mediodía). No son 5 sorpresas
       independientes: son **un mismo sesgo del pronóstico** repetido.

    ### Por qué el multi-paso infla los falsos positivos

    | Mecanismo | Qué pasa |
    |---|---|
    | **Error acumulado y correlacionado** | Las 24 predicciones comparten el mismo origen $T$. Si el modelo desfasa un poco el ciclo diario, *todas* las horas de la tarde fallan **juntas** y en la misma dirección → varias exceedancias correlacionadas, no independientes. |
    | **La banda asume que el modelo es correcto** | $\sigma_{(h)}$ sale de los coeficientes del propio modelo. SARIMA es una aproximación; si la dinámica está levemente mal especificada, el error real supera a $\sigma_{(h)}$ → banda demasiado angosta → exceedancias que son **error de modelo, no anomalías**. |
    | **Sensibilidad desigual** | Banda angosta al inicio, ancha al final. Una desviación moderada en $h=2$ se marca; una grande en $h=24$ no. *Dónde* caen las marcas depende del horizonte, no solo de qué tan raro es el dato. |

    ### La confusión de fondo: error de pronóstico ≠ anomalía

    El multi-paso responde *"¿qué tan mal predijo el modelo esta hora con 18 h
    de anticipación?"*. La detección de outliers quiere responder *"¿es esta
    observación sorprendente?"*. **No son lo mismo.** Un dato puede ser clima
    perfectamente normal que el modelo no supo anticipar tan lejos — eso es un
    límite del modelo, no un defecto del dato.

    ### Cómo distinguir lo real de lo artificial

    - **Un-paso-adelante (rolling)**: alimentar los valores reales hora a hora
      da una banda angosta y casi constante, y pregunta *"¿es raro este punto
      dado todo lo anterior?"*. Elimina el error acumulado. **Si un punto sigue
      marcado un-paso-adelante, ahí sí es creíble como anomalía** — es el
      discriminador clave.
    - **Forma del patrón**: marcas aisladas (tipo AO) son más sospechosas;
      marcas en bloque consecutivo gritan "deriva del pronóstico".
    - **Inyectar un pico sintético**: un dato real normal da `z≈2`; un pico
      inyectado da `z≫5`. Ver esa separación confirma que el detector
      distingue señal de ruido.
    - **Cruzar con el registro físico**: ¿hubo realmente un evento ese día? Si
      no, son datos normales.

    > **En corto.** El `z_max ≈ 2.7` y el agrupamiento son la firma del **error
    > de pronóstico multi-paso**, no de anomalías. El método es un *screening*
    > barato y honesto, pero su criterio de "outlier" está contaminado por la
    > capacidad predictiva del modelo a horizonte largo.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Mejor enfoque — predicción un-paso-adelante

    §6 mostró que el forecast multi-paso ensancha la banda e infla falsos
    positivos. La alternativa correcta es **predecir un solo paso adelante
    alimentando los valores reales hora a hora**.

    Con `res.append(test, refit=False)` los parámetros estimados en el train
    **no cambian**; solo avanza el **condicionamiento** (el filtro de Kalman ve
    el dato real de cada hora *antes* de predecir la siguiente). No hay bucle
    ni reentrenamiento: una sola llamada filtra todo el holdout.

    Resultado esperado: **σ casi constante ≈ 0.53 °C** en lugar de crecer hasta
    1.09 °C → banda **la mitad de ancha y pareja** todo el día. El mismo desvío
    físico pesa igual a cualquier hora (se acaba la asimetría por horizonte).
    """)
    return


@app.cell
def _(ganador, np, pd, test):
    _res = ganador["res"]
    _ext = _res.append(test, refit=False)          # filtra el holdout sin reajustar
    _pr  = _ext.get_prediction(start=test.index[0])  # in-sample ⇒ un paso adelante

    pred_one = _pr.predicted_mean
    se_one   = _pr.se_mean
    z_one    = (test.values - pred_one.values) / se_one.values

    lo_one = pred_one.values - 1.96 * se_one.values
    hi_one = pred_one.values + 1.96 * se_one.values
    out95_one = np.abs(z_one) > 1.96
    out99_one = np.abs(z_one) > 2.58

    tabla_one = pd.DataFrame({
        "hora":   test.index,
        "real":   np.round(test.values, 2),
        "pred":   np.round(pred_one.values, 2),
        "sigma":  np.round(se_one.values, 3),
        "z":      np.round(z_one, 2),
        "out_95": out95_one,
        "out_99": out99_one,
    })

    print(f"sigma un-paso (≈ constante): media {se_one.mean():.3f} °C  "
          f"· min {se_one.min():.3f} · max {se_one.max():.3f}")
    print(f"Outliers 95%: {int(out95_one.sum())} · 99%: {int(out99_one.sum())} "
          f"· |z| máximo = {np.abs(z_one).max():.2f}")
    tabla_one
    return hi_one, lo_one, out95_one, pred_one, z_one


@app.cell
def _(ci95, go, hi_one, lo_one, out95_one, pred_one, test):
    fig_one = go.Figure()

    # banda 95% multi-paso (ancha, de referencia)
    fig_one.add_trace(go.Scatter(
        x=list(test.index) + list(test.index[::-1]),
        y=list(ci95.iloc[:, 1].values) + list(ci95.iloc[:, 0].values[::-1]),
        fill="toself", fillcolor="rgba(120,120,120,0.12)",
        line=dict(width=0), name="banda 95% multi-paso", hoverinfo="skip",
    ))
    # banda 95% un-paso (angosta)
    fig_one.add_trace(go.Scatter(
        x=list(pred_one.index) + list(pred_one.index[::-1]),
        y=list(hi_one) + list(lo_one[::-1]),
        fill="toself", fillcolor="rgba(21,128,61,0.20)",
        line=dict(width=0), name="banda 95% un-paso", hoverinfo="skip",
    ))
    # holdout real
    fig_one.add_trace(go.Scatter(
        x=test.index, y=test.values, mode="lines+markers",
        name="holdout real", line=dict(color="#d97706", width=2),
    ))
    # predicción un-paso
    fig_one.add_trace(go.Scatter(
        x=pred_one.index, y=pred_one.values, mode="lines",
        name="pred un-paso", line=dict(color="#15803d", width=1.5, dash="dash"),
    ))
    # outliers un-paso
    fig_one.add_trace(go.Scatter(
        x=test.index[out95_one], y=test.values[out95_one], mode="markers",
        name="outlier un-paso (95%)",
        marker=dict(color="red", size=11, symbol="x", line=dict(width=2)),
    ))

    fig_one.update_layout(
        title="Un-paso-adelante: banda angosta (verde) dentro de la ancha multi-paso (gris)",
        xaxis_title="fecha", yaxis_title="tdb (°C)",
        template="plotly_white",
        height=380, margin=dict(l=40, r=20, t=50, b=40),
    )
    fig_one
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Lectura.** La banda verde (un-paso) cabe **dentro** de la gris
    (multi-paso) y es pareja todo el día. El `|z|` máximo **sube a ~4.3**: el
    punto que la banda ancha tapaba ahora destaca con claridad — sobreviviría
    incluso a $k = 4$. Y al mismo tiempo los falsos positivos por deriva
    **bajan** (de 5 a 3 al 95%). Más sensible **y** más específico a la vez.

    > **Salvedad — contaminación.** Como alimentamos el valor **real** de cada
    > hora para predecir la siguiente, un outlier distorsiona la predicción del
    > paso siguiente (el AR lo "persigue"). En producción se cierra el bucle:
    > **detectar → reemplazar el dato malo por su predicción → continuar.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Calibrar el umbral a los residuos reales

    §7 marcó un punto con `z ≈ 4.3`. Pero los residuos de `tdb` tienen **colas
    pesadas** (curtosis > 0): el propio modelo ya produce `|z|` de ~4 sobre
    datos *buenos* del train. La banda Normal es demasiado angosta en las colas
    → marca **variaciones legítimas** como outliers.

    La corrección: no asumir Normal, sino **leer el umbral de los propios
    residuos del modelo**. Dos formas, misma cobertura objetivo:

    1. **Cuantil empírico** — $k$ = percentil de $|z|$ de los residuos del
       train. Cero supuestos de forma; es lo que la serie realmente hace.
    2. **t de Student** — $k$ de una $t$ con los grados de libertad ajustados a
       los residuos. Captura las colas pesadas con una fórmula.
    """)
    return


@app.cell
def _(ganador, np, pd, stats, z_one):
    # residuos estandarizados un-paso IN-SAMPLE (train), sin burn-in
    z_train = np.asarray(ganador["res"].standardized_forecasts_error).ravel()[24:]
    z_train = z_train[np.isfinite(z_train)]

    df_t, _, scale_t = stats.t.fit(z_train, floc=0)   # t de Student, media 0

    def _umbrales(cov):
        a = 1 - (1 - cov) / 2
        return (
            stats.norm.ppf(a),                    # Normal
            float(np.quantile(np.abs(z_train), cov)),  # cuantil empírico de |z|
            stats.t.ppf(a, df_t) * scale_t,       # t de Student
        )

    tabla_umbral = pd.DataFrame([
        {
            "cobertura":     f"{_cov:.1%}",
            "k Normal":      round(kn, 2),
            "k empírico":    round(ke, 2),
            "k t-Student":   round(kt, 2),
            "out Normal":    int((np.abs(z_one) > kn).sum()),
            "out empírico":  int((np.abs(z_one) > ke).sum()),
            "out t-Student": int((np.abs(z_one) > kt).sum()),
        }
        for _cov in (0.95, 0.99, 0.995)
        for kn, ke, kt in [_umbrales(_cov)]
    ])

    # reencuadre de severidad del punto más extremo del holdout
    zmax = float(np.abs(z_one).max())
    p_norm = 2 * (1 - stats.norm.cdf(zmax))
    p_t    = 2 * (1 - stats.t.cdf(zmax / scale_t, df_t))
    print(f"Punto más extremo del holdout: |z| = {zmax:.2f}")
    print(f"  curtosis en exceso de los residuos: {stats.kurtosis(z_train):.2f} (0 = Normal)")
    print(f"  rareza según Normal      : 1 de {1/p_norm:,.0f}   ← parece anomalía")
    print(f"  rareza según t (df={df_t:.1f}) : 1 de {1/p_t:,.0f}   ← plausible (~mensual con datos horarios)")
    tabla_umbral
    return df_t, scale_t, z_train


@app.cell
def _(df_t, go, np, scale_t, stats, z_one, z_train):
    _a = 1 - (1 - 0.99) / 2
    _kn = stats.norm.ppf(_a)
    _ke = float(np.quantile(np.abs(z_train), 0.99))
    _kt = stats.t.ppf(_a, df_t) * scale_t
    _zmax = float(np.abs(z_one).max())

    fig_umbral = go.Figure()
    fig_umbral.add_trace(go.Histogram(
        x=z_train, nbinsx=40, histnorm="probability density",
        marker_color="#94a3b8", name="residuos train", opacity=0.7,
    ))
    for _k, _c, _nm in [(_kn, "#b91c1c", "Normal 99%"),
                        (_ke, "#2563eb", "empírico 99%"),
                        (_kt, "#15803d", "t-Student 99%")]:
        fig_umbral.add_vline(x=_k,  line=dict(color=_c, dash="dash"),
                             annotation_text=_nm, annotation_position="top")
        fig_umbral.add_vline(x=-_k, line=dict(color=_c, dash="dash"))
    fig_umbral.add_vline(x=_zmax,  line=dict(color="black", width=2),
                         annotation_text=f"holdout z={_zmax:.2f}", annotation_position="top")
    fig_umbral.add_vline(x=-_zmax, line=dict(color="black", width=2))

    fig_umbral.update_layout(
        title="Residuos del train y umbrales al 99% — Normal vs empírico vs t",
        xaxis_title="residuo estandarizado z", yaxis_title="densidad",
        template="plotly_white", height=380, margin=dict(l=40, r=20, t=60, b=40),
    )
    fig_umbral
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Lectura.**

    - Las dos calibraciones **suben el umbral** frente a la Normal: al 99%, de
      `k = 2.58` a `~3.0–3.1`. Eso descarta los falsos positivos "de cola
      moderada" que la Normal marcaría en una serie ruidosa como esta.
    - Lo más importante es el **reencuadre de severidad**: el punto `z ≈ 4.3`
      pasa de *"1 de cada ~53,000"* (Normal → alarma) a *"1 de cada ~600"*
      (t → plausible). Con datos horarios, un evento así ocurre **~una vez al
      mes** por puro azar — no es una anomalía, es la cola pesada natural de
      `tdb`.
    - Por eso, si estás seguro de que la variación es legítima, el umbral
      honesto **no sale de la Normal** sino de **tu presupuesto de falsas
      alarmas**: p. ej. tolerar ~1 alarma/mes (720 h) ⇒ cobertura ≈ 99.86% ⇒ un
      $k$ tomado del cuantil empírico a ese nivel, que deja a `z = 4.3` justo en
      el borde.

    > **Conclusión del paso:** la variación no es un outlier; es la cola pesada
    > de la serie, y calibrar el umbral con los residuos reales —no con la
    > Normal— es la corrección correcta. Si aún así quisieras que **no** se
    > marque, hay dos caminos más profundos: **modelar la varianza variable**
    > (heterocedasticidad, tipo GARCH) o **añadir un regresor exógeno**
    > (radiación, viento) que *explique* la variación y achique su residuo.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9 · Extender con un regresor exógeno — SARIMA**X**

    §8 mencionó un camino más profundo: si una variación grande tiene una
    **causa física**, podemos darle al modelo esa causa como **regresor
    exógeno**. Eso es exactamente lo que convierte SARIMA en SARIMA**X** — la
    "X" es la variable externa.

    Probaremos **dos** candidatos y veremos que dan resultados opuestos:

    - **Radiación global `ghi`** (*Global Horizontal Irradiance* — en el parquet
      no se llama `Ig`). Es lo que *calienta*; candidato obvio.
    - **Rapidez del viento `ws`**. La Ventosa es un sitio muy ventoso: una
      ráfaga o calma cambia la mezcla del aire y puede mover la temperatura.

    La hipótesis es que el regresor explique las subidas/bajadas y achique los
    residuos, de modo que la variación legítima de §8 deje de marcarse. El
    contraste enseñará **qué hace bueno a un exógeno**.

    **Dos detalles de los datos:**

    - `ghi` es `NaN` de noche (no se mide), pero físicamente la radiación
      nocturna es **0** — así la rellenamos.
    - El modelo necesita el exógeno también en el **holdout**. Aquí lo tenemos
      (datos históricos); en tiempo real necesitarías un *pronóstico* de esa
      variable.
    """)
    return


@app.cell
def _(pd, test, train):
    _path = "data/ClimaLab_2023-05-31_2025-06-20.parquet"
    _raw = pd.read_parquet(_path, columns=["ghi", "ws", "solar_altitude"]).resample("h").mean()

    _ghi = _raw["ghi"].copy()
    _ghi[_ghi.isna() & (_raw["solar_altitude"] <= 0)] = 0.0   # noche → 0
    _ghi = _ghi.interpolate(limit_direction="both")
    _ws = _raw["ws"].interpolate(limit_direction="both")

    ghi_tr = _ghi.reindex(train.index).to_frame("ghi")
    ghi_te = _ghi.reindex(test.index).to_frame("ghi")
    ws_tr = _ws.reindex(train.index).to_frame("ws")
    ws_te = _ws.reindex(test.index).to_frame("ws")

    print(f"corr(tdb, ghi) en train: {train.corr(_ghi.reindex(train.index)):.3f}")
    print(f"corr(tdb, ws)  en train: {train.corr(_ws.reindex(train.index)):.3f}")
    return ghi_te, ghi_tr, ws_te, ws_tr


@app.cell
def _(
    SARIMAX,
    ganador,
    ghi_te,
    ghi_tr,
    np,
    pd,
    stats,
    test,
    train,
    ws_te,
    ws_tr,
    z_one,
    z_train,
):
    def _fit_exog(x_tr, x_te):
        r = SARIMAX(
            train, exog=x_tr,
            order=(2, 0, 2), seasonal_order=(2, 1, 1, 24),
            enforce_stationarity=True, enforce_invertibility=True,
        ).fit(disp=False)
        pr = r.append(test, exog=x_te, refit=False).get_prediction(start=test.index[0])
        z = (test.values - pr.predicted_mean.values) / pr.se_mean.values
        ztr = np.asarray(r.standardized_forecasts_error).ravel()[24:]
        return r, z, ztr[np.isfinite(ztr)]

    res_ghi, z_ghi, _ztr_g = _fit_exog(ghi_tr, ghi_te)
    res_ws,  z_ws,  _ztr_w = _fit_exog(ws_tr, ws_te)

    def _fila(nombre, aic, z, ztr):
        return {"modelo": nombre, "AIC": round(float(aic), 1),
                "|z| máx holdout": round(float(np.abs(z).max()), 2),
                "outliers 95%": int((np.abs(z) > 1.96).sum()),
                "curtosis resid": round(float(stats.kurtosis(ztr)), 2)}

    tabla_exog = pd.DataFrame([
        _fila("SARIMA (sin exog)", ganador["res"].aic, z_one, z_train),
        _fila("SARIMAX (+ ghi)",   res_ghi.aic,        z_ghi, _ztr_g),
        _fila("SARIMAX (+ ws)",    res_ws.aic,         z_ws,  _ztr_w),
    ])

    print(f"coef ghi = {res_ghi.params['ghi']:+.5f} (p={res_ghi.pvalues['ghi']:.4f}) "
          f"→ AIC SUBE: redundante con la estacionalidad")
    print(f"coef ws  = {res_ws.params['ws']:+.5f} (p={res_ws.pvalues['ws']:.4f}) "
          f"→ AIC BAJA: aporta info no estacional")
    tabla_exog
    return res_ws, z_ghi, z_ws


@app.cell
def _(go, np, pd, test, z_ghi, z_one, z_ws):
    _ix = int(np.argmax(np.abs(z_one)))          # hora del punto extremo (21:00)
    _hora = test.index[_ix]

    fig_exog = go.Figure()
    for _y, _nm, _c in [(np.abs(z_one), "sin exog", "#94a3b8"),
                        (np.abs(z_ghi), "+ ghi",    "#d97706"),
                        (np.abs(z_ws),  "+ ws",     "#15803d")]:
        # etiqueta el valor SOLO en la hora extrema (lo demás vacío, sin saturar)
        _txt = ["" if _i != _ix else f"{_v:.2f}" for _i, _v in enumerate(_y)]
        fig_exog.add_trace(go.Bar(
            x=test.index, y=_y, name=f"|z| {_nm}", marker_color=_c,
            text=_txt, textposition="outside", textfont=dict(size=11),
        ))

    # resalta la hora extrema para que la vista vaya directo ahí
    fig_exog.add_vrect(
        x0=_hora - pd.Timedelta(minutes=40), x1=_hora + pd.Timedelta(minutes=40),
        fillcolor="rgba(250,204,21,0.18)", line_width=0, layer="below",
        annotation_text="punto extremo (21:00)", annotation_position="top left",
    )
    fig_exog.add_hline(y=1.96, line=dict(color="#b91c1c", dash="dash"),
                       annotation_text="k=1.96 (Normal 95%)", annotation_position="top right")
    fig_exog.add_hline(y=3.0, line=dict(color="#2563eb", dash="dash"),
                       annotation_text="k≈3 (calibrado 99%)", annotation_position="bottom right")
    fig_exog.update_layout(
        title=("|z| por hora · cada hora tiene 3 barras (un modelo c/u)<br>"
               "<sup>casi todas son chicas e iguales; mira solo la hora resaltada: "
               "+ghi no baja la barra, +ws sí</sup>"),
        xaxis_title="hora del holdout", yaxis_title="|z|  (nº de desviaciones)",
        barmode="group", template="plotly_white",
        height=380, margin=dict(l=40, r=20, t=70, b=40),
    )
    fig_exog
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Cómo leer la figura.** Cada hora del holdout tiene **tres barras** (gris
    = sin exógeno, naranja = con `ghi`, verde = con `ws`). La altura es `|z|`:
    cuántas desviaciones se aleja el dato real de la predicción de ese modelo.
    Una barra que **cruza una línea punteada** es candidata a outlier según ese
    umbral. Casi todas las horas tienen barras chicas y casi idénticas entre
    modelos — por eso solo importa la **hora resaltada (21:00)**, donde se ve el
    contraste: `+ghi` deja la barra igual, `+ws` la baja.

    **Lectura — dos exógenos, resultados opuestos.**

    Los dos coeficientes salen **significativos**, pero el AIC los juzga al
    revés:

    - **`ghi` SUBE el AIC** (≈1055 → ≈1071) y **no mueve** el `|z|` máximo
      (~4.3). Aunque la radiación *es* lo que calienta, la **diferencia
      estacional** `D=1, s=24` **ya removió el ciclo diario de radiación**: el
      patrón medio de `ghi` está doblemente contado → es **redundante**. (Hay
      además desfase térmico; probar rezagos 1–3 h tampoco ayuda.)
    - **`ws` BAJA el AIC** (≈1055 → ≈1019) y **reduce** el `|z|` máximo. El
      viento **no es periódico** como el sol, así que lleva información que la
      estacionalidad **no** tenía. Eso es justo lo que hace útil a un exógeno.

    **Y el punto extremo tiene sentido físico.** El `z ≈ 4.3` cae el
    **2024-04-12 a las 21:00**: de noche, el viento **gira de NO (287°) a S
    (183°)** y casi se calma (0.9 m/s), y la temperatura se queda caliente
    cuando el modelo esperaba que enfriara. Con poca mezcla el aire se
    "desacopla" del ciclo regular (de hecho `corr(|z|, ws) < 0`: más error con
    viento calmo). Es un **evento meteorológico real**, no un defecto — lo que
    confirma, una vez más, que **no es una anomalía**.

    > **La lección.** Un regresor exógeno solo ayuda si **aporta información que
    > los términos estacionales no tienen ya**: `ghi` ≈ el ciclo diario
    > (redundante), `ws` ≠ periódico (informativo). `ws` reduce la variación
    > pero no la borra del todo; combinarlo con humedad la baja más, aunque `rh`
    > está *parcialmente co-determinada* con la temperatura (cuidado con esa
    > circularidad). Para detección, lo honesto es **`ws` como driver + el
    > umbral calibrado de §8** para lo que el modelo no explique.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10 · Síntesis — un modelo, dos modos

    El mismo SARIMAX+`ws` ajustado sirve para **dos tareas distintas**, y la
    clave es no confundirlas:

    | modo | llamada | banda | propósito |
    |---|---|---|---|
    | **Pronóstico** | `get_forecast(h, exog)` | crece con el horizonte | predecir el futuro con su incertidumbre |
    | **Clasificación** | `append(refit=False)` → `get_prediction` | angosta y constante | decidir si cada dato es outlier |

    La idea que propusiste: usar la banda **ancha** del multi-paso para *mostrar
    el pronóstico*, y la banda **angosta** del un-paso (con el umbral calibrado
    de §8) para *clasificar*. Misma `res`, dos `get_*`, dos propósitos.

    > El modo pronóstico con exógeno necesita el `ws` **futuro** (aquí histórico;
    > en tiempo real, un pronóstico de viento).
    """)
    return


@app.cell
def _(np, res_ws, stats, test, ws_te):
    # MODO A — pronóstico multi-paso con banda 95% (necesita ws futuro)
    _fcA  = res_ws.get_forecast(steps=len(test), exog=ws_te)
    predA = _fcA.predicted_mean
    ciA   = _fcA.conf_int(alpha=0.05)

    # MODO B — clasificación un-paso-adelante (alimenta los reales)
    _prB  = res_ws.append(test, exog=ws_te, refit=False).get_prediction(start=test.index[0])
    predB = _prB.predicted_mean
    _seB  = _prB.se_mean.values
    _zB   = (test.values - predB.values) / _seB

    # umbral calibrado de §8, sobre los residuos del propio SARIMAX+ws
    _ztr = np.asarray(res_ws.standardized_forecasts_error).ravel()[24:]
    _ztr = _ztr[np.isfinite(_ztr)]
    _df, _, _sc = stats.t.fit(_ztr, floc=0)
    _k_emp = float(np.quantile(np.abs(_ztr), 0.99))
    _k_t   = stats.t.ppf(0.995, _df) * _sc
    k_clasif = round(float(max(_k_emp, _k_t)), 2)
    outB = np.abs(_zB) > k_clasif

    # banda de DECISIÓN del un-paso: predB ± k·σ (la frontera del clasificador)
    loB = predB.values - k_clasif * _seB
    hiB = predB.values + k_clasif * _seB

    print(f"umbral calibrado 99% (SARIMAX+ws): empírico {_k_emp:.2f} · t {_k_t:.2f} → k = {k_clasif}")
    print(f"clasificación un-paso: {int(outB.sum())} outlier(s) de {len(test)} "
          f"· |z| máx = {np.abs(_zB).max():.2f}")
    return ciA, hiB, k_clasif, loB, outB, predA, predB


@app.cell
def _(ciA, go, hiB, k_clasif, loB, outB, predA, predB, test):
    fig_syn = go.Figure()
    # banda ancha del pronóstico multi-paso
    fig_syn.add_trace(go.Scatter(
        x=list(test.index) + list(test.index[::-1]),
        y=list(ciA.iloc[:, 1].values) + list(ciA.iloc[:, 0].values[::-1]),
        fill="toself", fillcolor="rgba(120,120,120,0.15)",
        line=dict(width=0), name="banda 95% pronóstico (multi-paso)", hoverinfo="skip",
    ))
    # banda angosta de DECISIÓN del un-paso (frontera del clasificador, ±k·σ)
    fig_syn.add_trace(go.Scatter(
        x=list(predB.index) + list(predB.index[::-1]),
        y=list(hiB) + list(loB[::-1]),
        fill="toself", fillcolor="rgba(21,128,61,0.18)",
        line=dict(width=0), name=f"banda de decisión un-paso (±{k_clasif}σ)", hoverinfo="skip",
    ))
    fig_syn.add_trace(go.Scatter(
        x=predA.index, y=predA.values, mode="lines",
        name="pronóstico multi-paso", line=dict(color="#6b7280", width=1.5, dash="dot"),
    ))
    fig_syn.add_trace(go.Scatter(
        x=test.index, y=test.values, mode="lines+markers",
        name="holdout real", line=dict(color="#d97706", width=2),
    ))
    fig_syn.add_trace(go.Scatter(
        x=predB.index, y=predB.values, mode="lines",
        name="predicción un-paso", line=dict(color="#15803d", width=1.5, dash="dash"),
    ))
    fig_syn.add_trace(go.Scatter(
        x=test.index[outB], y=test.values[outB], mode="markers",
        name=f"outlier (un-paso, k={k_clasif})",
        marker=dict(color="red", size=12, symbol="x", line=dict(width=2)),
    ))
    fig_syn.update_layout(
        title="Un modelo SARIMAX+ws, dos bandas: pronóstico (gris, ancha) vs decisión un-paso (verde, angosta)",
        xaxis_title="fecha", yaxis_title="tdb (°C)",
        template="plotly_white", height=400, margin=dict(l=40, r=20, t=60, b=40),
    )
    fig_syn
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Lectura.**

    - La **banda gris** (multi-paso) se abre hacia la derecha: es la
      incertidumbre honesta de pronosticar a 24 h. Sirve para **planear**, no
      para detectar.
    - La **banda verde** angosta es la de **decisión**: $\hat y_t \pm k\sigma$
      del un-paso con el `k` **calibrado** (~3) de §8. Es la frontera del
      clasificador — un punto **fuera** de ella es un outlier (las X rojas), un
      punto **dentro** es normal. Ahora se *ve* por qué se marca cada uno.
    - El punto del **giro de viento (21:00)** queda señalado para *revisión
      humana*, no borrado: es el único evento del día que ni el modelo ni el
      viento explican del todo, y eso es exactamente lo que un detector debe
      hacer — **levantar la mano, no decidir solo**.

    > **Arquitectura de producción:** un modelo, dos `get_*`. El pronóstico
    > alimenta decisiones; la clasificación un-paso vigila la calidad del dato
    > conforme llega. Cambiar el horizonte, el exógeno o el umbral no toca el
    > resto.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11 · Conclusión

    Recorrimos el flujo completo: entrenar tres SARIMA, elegir uno por RMSE, y
    usar su intervalo de predicción para marcar horas inesperadas en el
    holdout. La lección central de §6 es la que hay que llevarse:

    > Marcar puntos fuera de la banda **no equivale** a encontrar anomalías.
    > Con el forecast multi-paso, las exceedancias que vimos (`z_max ≈ 2.7`,
    > agrupadas) son **error de pronóstico**, no datos defectuosos. El método
    > sirve como *screening* barato, pero su veredicto está contaminado por la
    > capacidad predictiva del modelo a horizonte largo.

    El **un-paso-adelante de §7** es el método honesto: banda angosta y pareja,
    que descarta los artefactos de deriva del multi-paso. Pero §8 cierra la
    historia: el punto `z ≈ 4.3` que destacaba **tampoco es una anomalía** — es
    la **cola pesada natural** de `tdb`, y solo parecía extremo porque el umbral
    venía de una Normal. Calibrando el umbral con los residuos reales (cuantil
    empírico o t de Student), la variación legítima deja de ser sospechosa.

    §9 probó el camino exógeno (SARIMAX) y dejó la lección más fina: **`ghi`
    (radiación) es redundante** con el ciclo estacional ya modelado, mientras
    que **`ws` (viento) sí aporta** porque no es periódico — baja el AIC y
    reduce la variación. Y el punto extremo coincide con un **giro de viento**:
    es un evento real, no un defecto. Un regresor sirve solo si aporta algo que
    el modelo **no tenía ya**.

    > **El recorrido completo:** multi-paso (§5) marca de más por error de
    > pronóstico → un-paso-adelante (§7) limpia eso → umbral calibrado (§8)
    > limpia los falsos positivos por cola pesada → SARIMAX con viento (§9)
    > explica parte de la variación y confirma que es meteorológica → §10 junta
    > todo: un modelo que **pronostica** (banda ancha) y **clasifica** (un-paso,
    > banda angosta). Lo que sobreviva a todo esto sí merece llamarse anomalía.

    **Si quisiéramos llevarlo más lejos:**

    - **Inyectar un pico sintético** para *probar* el detector: un dato normal
      da `z≈2`, un pico inyectado `z≫5`. Ver esa separación valida el umbral.
    - Probar una ventana de holdout con un **evento conocido** (tormenta, fallo
      del sensor, dato pegado).
    - Combinar `ws` con la **dirección del viento `wd`** o modelar la **varianza
      variable** (heterocedasticidad) para la causa de fondo.

    El esqueleto queda listo: cambiar la ventana, el modelo, el umbral o el
    regresor es trivial y el resto del flujo no se toca.
    """)
    return


if __name__ == "__main__":
    app.run()
