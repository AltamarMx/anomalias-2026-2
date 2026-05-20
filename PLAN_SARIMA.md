# PLAN — Aprender SARIMA con `tdb` (ClimaLab)

Una secuencia de libretas marimo cortas e interactivas que combinan teoría y
datos, usando **solo la columna `tdb`** (temperatura de bulbo seco) del
parquet `data/ClimaLab_2023-05-31_2025-06-20.parquet`.

## Objetivos

1. **Entender y configurar SARIMA viendo los datos.**
2. **Pronosticar** con SARIMA y evaluar la calidad del pronóstico.
3. **Detectar anomalías** con SARIMA usando residuos / intervalos.

## Datos

- 1,076,768 observaciones a ~1 minuto, del 2023-05-31 al 2025-06-20.
- Sin huecos significativos: 99.98% de los pasos son de 1 min exacto.
- Rango: 10.98 °C – 39.93 °C; media 24.45 °C; sin NaNs.
- **Dos estacionalidades evidentes**: diaria (s=24 si remuestreamos a hora,
  s=144 si remuestreamos a 10 min) y anual.
- Decisión por defecto: **remuestrear a hora** (`resample("h").mean()`) para
  trabajar con SARIMA diario `s=24` cuando interese; cortar ventanas
  manejables (días–semanas) para no pelearse con el tamaño.

## Stack común en cada libreta

- `marimo` con celdas reactivas + widgets (`mo.ui.slider`, `mo.ui.range_slider`,
  formularios).
- **plotly** para gráficas interactivas (zoom, hover, panning). Falta agregar
  `plotly` a `pyproject.toml`; lo haremos en la libreta 1 con
  `ctx.packages.add("plotly")`.
- `statsmodels.tsa.statespace.SARIMAX` para el modelo.
- Helpers compartidos en una celda inicial (carga del parquet, slicing por
  fecha, train/test split, métricas) — pueden empezar locales por libreta y
  luego, si se repiten mucho, extraerse a `notebooks/_sarima_utils.py`.

## Libretas propuestas

### `019_tdb_exploracion.py` — *Introducción a SARIMA y exploración de la serie*

**Idea.** Doble propósito en una sola libreta:

1. **Presentar SARIMA** — fórmula y parámetros, sin ajustar nada todavía. 
Como se interpreta la formula, como se construyó así ?
2. **Conocer la serie por dentro** — estacionalidades, tendencia, ruido,
   estacionariedad — para que cada parámetro de SARIMA tenga una imagen
   concreta del fenómeno que está capturando.

**Contenido.**

#### Parte A — Qué es SARIMA (teoría)
1. **De AR a SARIMA en una tabla**. Mostrar la jerarquía
   AR → MA → ARMA → ARIMA → SARIMA como casos particulares de una sola
   ecuación, con la notación `(p, d, q)(P, D, Q, s)` y qué pasa al poner
   ceros.
2. **La fórmula compacta de SARIMA** con el operador de rezago `L`:

   $$
   \Phi_P(L^{s})\,\phi_p(L)\,(1-L^{s})^{D}\,(1-L)^{d}\,y[t]
   \;=\;
   \Theta_Q(L^{s})\,\theta_q(L)\,\varepsilon[t]
   $$

   Leerla de derecha a izquierda: primero diferenciar, después AR y MA
   regulares y estacionales.
3. **Parámetro a parámetro** — qué hace cada uno y cómo se relaciona con la
   serie `tdb`:
   - `p, q` — dinámica de minutos/horas inmediatas.
   - `d` — quitar tendencia (¿la hay en `tdb`?).
   - `P, Q` — memoria de "mismo instante de ciclos anteriores".
   - `D` — diferenciación estacional (`y[t] - y[t-s]`).
   - `s` — periodo del ciclo. Para `tdb` hourly, `s = 24`.
4. **Aclaración importante**: `D = 1` con `s = 24` **no calcula un perfil
   promedio horario** — hace `y[t] - y[t-24]`. El perfil queda implícito.
5. **Por qué se multiplican** las partes regular y estacional — aparecen
   rezagos cruzados (1, 24, 25) "gratis".

> Esta parte teórica se apoya en lo que ya está en `017_SARIMA.py`, pero
> resumida y con ejemplos en términos de temperatura, no viento.

#### Parte B — Explorar `tdb` con esos parámetros en mente
6. Cargar `tdb`, mostrar tamaño y rango. Selector de ventana
   (`mo.ui.date_range`) y de frecuencia de remuestreo
   (`mo.ui.radio` con minuto / 10-min / hora / día).
7. **Plot interactivo** plotly de la serie completa + ventana seleccionada
   — pregunta guía: *¿qué `s` salta a la vista?*
8. **Perfil diario** (boxplot/heatmap por hora del día) y **perfil anual**
   (heatmap mes × día u hora × día del año) — pregunta guía: *¿hay
   estacionalidad anual además de la diaria?*
9. **ACF / PACF** sobre una ventana hourly de varias semanas — relacionar
   los picos con `p`, `q`, `P`, `Q` directamente.
10. **Estacionariedad**: pruebas ADF y KPSS antes y después de aplicar
    `diff()`, `diff(24)`, ambas. Conectar con `d` y `D`: ¿qué pareja
    `(d, D)` deja la serie estacionaria?
11. Cierre: **¿Qué `(p, d, q)(P, D, Q, s)` esperarías intentar primero
    en la libreta 020?** Conclusiones que alimentan la siguiente libreta.

---

### `020_tdb_configuracion.py` — *Configurar SARIMA con el widget*

**Idea.** Lo que ya hace `017_SARIMA.py` con viento de La Ventosa, pero ahora
con `tdb` y con plotly. El énfasis es **leer los diagnósticos** (residuos,
ACF/PACF de residuos, Ljung-Box) para iterar las perillas
`(p, d, q)(P, D, Q, s)`.

**Contenido.**
1. Recordar la ecuación SARIMA (resumen breve, no repetir todo lo de 017;
   un cuadro tabla + una imagen mental).
2. Selector de ventana de entrenamiento (varias semanas hourly), separar
   train/test (`mo.ui.number` para el horizonte en horas).
3. **Widget de órdenes** `(p,d,q)(P,D,Q,s)` con `s=24` por defecto.
4. Fit + tabla AIC/BIC + RMSE/MAE en holdout.
5. **Panel de diagnóstico plotly 2×2**: forecast vs holdout, residuos, ACF y
   PACF de residuos. Hover muestra el lag exacto.
6. **Prueba de Ljung-Box** sobre residuos como semáforo de "ya no hay
   estructura".
7. Recetario corto: "si ves esto en residuos → mueve esta perilla".

**Teoría incluida.** Cómo leer ACF/PACF de residuos, qué dice Ljung-Box,
trade-off AIC vs RMSE.

---

### `021_tdb_pronostico.py` — *Pronosticar con SARIMA*

**Idea.** Saltar de "ajustar un modelo" a "hacerle pronósticos útiles" —
horizontes cortos, medianos, y cómo se degrada el pronóstico.

**Contenido.**
1. Fijar un modelo "razonable" elegido en la libreta 020 (lo dejamos
   parametrizable con sliders pero con defaults concretos).
2. **Pronóstico h pasos adelante** con `get_forecast(steps=h)`; slider para
   `h` entre 1 y 168 horas.
3. **Intervalos de confianza** al 80% y 95% — visualizar cómo se abren con
   el horizonte; explicar por qué.
4. **Rolling forecast / walk-forward**: cada ventana de N horas, refit (o
   `apply()`) y predecir 1 paso. Comparar RMSE one-step contra h-step.
5. **Baselines**: comparar el SARIMA contra
   (a) persistencia (`y_hat[t+1] = y[t]`) y
   (b) "ayer a la misma hora" (`y_hat[t] = y[t-24]`). Una tabla con RMSE de
   los tres da perspectiva.
6. Discusión final: ¿en qué horizonte deja SARIMA de ser útil para `tdb`?

**Teoría incluida.** Diferencia entre pronóstico in-sample y out-of-sample,
qué representa el intervalo de confianza, por qué crece con el horizonte.

---

### `022_tdb_anomalias.py` — *Detectar anomalías con SARIMA*

**Idea.** Usar el modelo entrenado como **referencia de normalidad**: lo que
el modelo no podía predecir bien y se sale del intervalo es candidato a
anomalía.

**Contenido.**
1. Recapitular el modelo elegido y refitearlo en un periodo "limpio" (slider
   para elegirlo).
2. **Pronóstico one-step-ahead** sobre todo el periodo con `get_prediction()`
   → residuos en cada paso.
3. **Score de anomalía** = residuo estandarizado (`resid / sigma`). Slider
   para umbral en sigmas (e.g. 3σ, 4σ) o percentil empírico.
4. **Plotly interactivo**: serie con puntos rojos marcando anomalías; hover
   muestra valor real, predicho, residuo, score.
5. **Anomalías estacionales**: separar "frío anormal a las 14:00" de "frío
   normal a las 04:00" — el modelo ya hace esto implícitamente, hay que
   demostrarlo con un ejemplo.
6. **Ventana streaming**: simular llegada en tiempo real, ir actualizando
   con `res.append()` para ver cómo el modelo se adapta o no a un cambio
   de régimen.
7. Caveat: SARIMA marca *desviaciones del patrón aprendido*, no
   necesariamente "errores físicos" — discutir cuándo conviene SARIMA vs
   reglas de rango fijo.

**Teoría incluida.** Residual como detector; relación entre intervalo de
confianza y umbral; falsos positivos vs falsos negativos según el umbral.

---

## Lo que **no** está en este plan (a decidir)

- ¿Libreta extra de **selección automática de órdenes** (`pmdarima.auto_arima`
  o búsqueda en cuadrícula con AIC)? Útil pero opcional.
- ¿Tocar **estacionalidad anual** (s=8760 hourly o s=365 diario)? Costoso de
  ajustar; quizá solo mencionarlo en la libreta 1 y dejarlo para más
  adelante.
- ¿Comparar SARIMA contra **STL + ARMA en residuos** o **Prophet**? Sería
  una libreta extra de "alternativas".
- ¿Una libreta de **utilidades compartidas** (`_sarima_utils.py`) si se
  vuelve pesado duplicar código?

## Cosas que confirmar antes de empezar

1. **Frecuencia base por defecto: hora.** ¿De acuerdo, o prefieres 10 min?
2. **Ventana de trabajo.** Para SARIMA hourly con `s=24`, un mes (≈720 obs)
   ya es bastante; ¿prefieres trabajar siempre con el mismo mes o que cada
   libreta tenga selector libre?
3. **Plotly como librería obligatoria** — agregarla a `pyproject.toml`.
4. **Nombres de archivo**: `019_…`, `020_…`, `021_…`, `022_…` siguiendo la
   numeración existente. ¿Te parece, o quieres prefijo distinto
   (`019_SARIMA_*`)?
