# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # 015 · ARMA Cheatsheet — Box-Jenkins de bolsillo
#
# **Estadística, Detección de Anomalías e Imputación de Series Temporales**
# Posgrado en Ingeniería · Área Energía · IER-UNAM
#
# Resumen de los conceptos, fórmulas y decisiones del laboratorio
# `014b_LaVentosa.py`. Pensado como referencia rápida — los ejemplos físicos y
# discusiones detalladas viven en el lab.

# %% [markdown]
# ## 1 · El ciclo Box-Jenkins
#
# Cuatro pasos en loop:
#
# | Paso          | Herramientas             | Decisión                                  |
# |---------------|--------------------------|-------------------------------------------|
# | Identificar   | ADF + ACF + PACF         | ¿Estacionaria? ¿Qué orden $(p, q)$ probar? |
# | Estimar       | `ARIMA(p, d, q).fit()`   | MLE por filtro de Kalman                  |
# | Diagnosticar  | ACF/PACF residual + LB   | ¿Residuales blancos?                      |
# | Decidir       | LB + AIC + parsimonia    | Subir orden o parar y pronosticar         |

# %% [markdown]
# ## 2 · Modelos AR / MA / ARMA
#
# Sobre serie **estacionaria**:
#
# - **AR(p)** — inercia (memoria del pasado observado):
#
# $$y_t = c + \sum_{i=1}^{p} \phi_i\, y_{t-i} + \varepsilon_t.$$
#
# - **MA(q)** — eco de choques (memoria de innovaciones):
#
# $$y_t = \mu + \varepsilon_t + \sum_{j=1}^{q} \theta_j\, \varepsilon_{t-j}.$$
#
# - **ARMA(p, q)** — combinación:
#
# $$y_t = c + \sum_{i=1}^{p} \phi_i\, y_{t-i} + \varepsilon_t + \sum_{j=1}^{q} \theta_j\, \varepsilon_{t-j}.$$
#
# - $p$ = rezagos del pasado **observado**.
# - $q$ = rezagos de **innovaciones** $\varepsilon_t$.
# - $\varepsilon_t$ = **ruido blanco**: media 0, varianza $\sigma^2$ constante, sin
#   correlación entre lags.

# %% [markdown]
# ## 3 · Identificación visual — ACF vs PACF
#
# **Reglas mnemotécnicas:**
#
# | Modelo  | ACF                | PACF              |
# |---------|--------------------|-------------------|
# | AR(p)   | decae              | **corta en $p$**  |
# | MA(q)   | **corta en $q$**   | decae             |
# | ARMA    | decae              | decae             |
#
# **Banda de confianza** (ruido bajo $H_0$): $\pm 1.96 / \sqrt{n}$.
#
# *"Cortar"* = primer lag claramente dentro de banda. Los picos significativos son
# **cota superior** del orden — empieza humilde y sube si los diagnósticos lo piden.
#
# > $p, q$ los lees del ACF/PACF (cuántos parámetros). Los valores $\phi_i, \theta_j$
# > los pone el optimizador después por MLE.

# %% [markdown]
# ## 4 · Raíz unitaria y estacionariedad
#
# **Polinomio característico** AR(p):
#
# $$1 - \phi_1\, z - \phi_2\, z^2 - \cdots - \phi_p\, z^p = 0.$$
#
# | Raíces $z_i$                | Comportamiento          |
# |------------------------------|--------------------------|
# | Todas $\lvert z_i\rvert > 1$ | **Estacionario**         |
# | Alguna $\lvert z_i\rvert = 1$| **Raíz unitaria** (no estacionario, paseo aleatorio) |
# | Alguna $\lvert z_i\rvert < 1$| Explosivo                |
#
# **Ejemplo de despeje (AR(2) con raíz unitaria):**
#
# $$y_t = 1.5\, y_{t-1} - 0.5\, y_{t-2} + \varepsilon_t
# \;\Longrightarrow\; 1 - 1.5\, z + 0.5\, z^2 = (1-z)(1-0.5z) = 0
# \;\Longrightarrow\; z_1 = 1,\; z_2 = 2.$$
#
# Diferenciar $\Delta y_t = y_t - y_{t-1}$ remueve la raíz unitaria → equivale a
# ARIMA(1, 1, 0).

# %% [markdown]
# ## 5 · ADF — Augmented Dickey-Fuller
#
# **Regresión:**
#
# $$\Delta y_t = \alpha + \beta\, t + \gamma\, y_{t-1} + \sum_{i=1}^{k} \delta_i\, \Delta y_{t-i} + \varepsilon_t,$$
#
# con $\gamma = \phi - 1$.
#
# - $H_0$: $\gamma = 0$ → **raíz unitaria** (NO estacionaria).
# - $H_1$: $\gamma < 0$ → **estacionaria**.
# - Regla: $p < 0.05$ → rechaza $H_0$.
#
# ⚠ **Necesario pero no suficiente para ARMA**. ADF es ciego a estacionalidad
# determinista (un ciclo periódico puede pasar el ADF y aún arruinar el ARMA, o
# confundir al test y darle un "no estacionaria" falso). Siempre inspecciona ACF y
# climatología.

# %% [markdown]
# ## 6 · Ljung-Box — blancura de residuales
#
# **Residual:** $\hat{\varepsilon}_t = y_t - \hat{y}_t$, la estimación empírica de la
# innovación. Si el modelo es adecuado, deben verse como ruido blanco.
#
# **Estadístico** (LB = Ljung-Box, Greta M. Ljung & George E. P. Box, 1978):
#
# $$Q_{LB}(h) = n(n+2) \sum_{k=1}^{h} \frac{\hat{\rho}_k^2}{n - k} \;\sim\; \chi^2_{h - p - q}.$$
#
# - $H_0$: residuales son ruido blanco.
# - $H_1$: queda autocorrelación.
# - Regla: $p > 0.05$ → **NO rechaza** → modelo adecuado.
#
# **Lags típicos para datos horarios:** $h = 10$ (media jornada) y $h = 20$ (día). Si
# rechaza, queda estructura → subir orden o cambiar de familia.

# %% [markdown]
# ## 7 · AIC y BIC — comparar modelos
#
# $$\text{AIC} = -2\log L_{\max} + 2k, \qquad \text{BIC} = -2\log L_{\max} + k\log n.$$
#
# **Menor es mejor.** Solo comparable entre modelos sobre **la misma serie**.
#
# | Diferencia          | Interpretación                                              |
# |---------------------|-------------------------------------------------------------|
# | $\Delta\text{AIC} > 2$ | Significativa                                            |
# | $\Delta\text{AIC} > 10$ | Decisiva                                                |
#
# - **AIC** penaliza con $+2k$ — más permisivo con la complejidad.
# - **BIC** penaliza con $+k \log n$ — más conservador cuando $n$ es grande.
#
# > Ljung-Box dice si un modelo es **adecuado** (sí/no). AIC dice cuál es **mejor**
# > entre los adecuados (ranking). Son complementarios — nunca elijas por AIC bajo
# > si LB rechaza.

# %% [markdown]
# ## 8 · Pitfalls y reglas operativas
#
# - **ADF dice "estacionaria" ≠ se puede ARMA.** Inspecciona también ACF y
#   climatología para descartar estacionalidad determinista.
# - **ACF residual con picos a múltiplos de $s$** → estacionalidad pendiente. ARMA no
#   la modela. Resta climatología o pasa a SARIMA$(p,d,q)(P,D,Q)_s$.
# - **AR con $\phi$ cerca de 1** → casi raíz unitaria. Comparar con ARIMA$(p, 1, q)$.
# - **Residuales con autocorrelación** → IC del pronóstico **subestimados**: el
#   modelo miente sobre su incertidumbre.
# - **AIC bajo + LB rechaza = modelo malo disfrazado.** El diagnóstico de blancura
#   manda sobre el ranking.
# - **El diagnóstico estadístico NO garantiza superioridad operativa.** Siempre
#   valida con **backtest** contra un baseline (persistencia, climatología sola, NWP).
# - **Parsimonia:** si dos modelos pasan LB con AIC parecido, gana el más simple.
# - **Cobertura del IC ≠ utilidad del pronóstico.** Un IC al 100% puede venir de
#   varianza incondicional inflada — útil para reportar incertidumbre, no para
#   decidir operación.

# %% [markdown]
# ## 9 · Snippets canónicos
#
# Imports estándar:
#
# ```python
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from statsmodels.stats.diagnostic import acorr_ljungbox
# ```
#
# **Identificación.**
#
# ```python
# # ADF
# stat, p_adf, *_ = adfuller(y, autolag="AIC")
#
# # ACF / PACF
# plot_acf(y,  lags=30)
# plot_pacf(y, lags=30, method="ywm")
# ```
#
# **Estimación.**
#
# ```python
# modelo = ARIMA(y, order=(p, d, q)).fit()
# print(modelo.summary())   # coef, std err, z, p-value, AIC, BIC
# ```
#
# **Diagnóstico.**
#
# ```python
# resid = modelo.resid.iloc[max(p, q):]
# plot_acf(resid, lags=30)
# plot_pacf(resid, lags=30, method="ywm")
# lb = acorr_ljungbox(resid, lags=[10, 20], return_df=True)
# ```
#
# **Pronóstico simple ($H$ pasos hacia el futuro).**
#
# ```python
# fc       = modelo.get_forecast(steps=H)
# fc_mean  = fc.predicted_mean
# fc_ic95  = fc.conf_int(alpha=0.05)
# ```
#
# **Backtest walk-forward (lo importante para validar).**
#
# ```python
# y_train = y.iloc[:-H]
# y_test  = y.iloc[-H:]
# modelo_eval = ARIMA(y_train, order=(p, d, q)).fit()
# fc = modelo_eval.get_forecast(steps=H)
#
# real = y_test.values
# pred = fc.predicted_mean.values
# mae  = np.abs(real - pred).mean()
# rmse = np.sqrt(((real - pred) ** 2).mean())
#
# # Baseline ingenuo
# mae_persist = np.abs(real - y.iloc[-2*H:-H].values).mean()
# ```

# %% [markdown]
# ## 10 · Helpers reutilizables (del lab)
#
# ```python
# def mirar(serie, nombre, lags=30):
#     """Identificación: serie + ADF + ACF + PACF."""
#     serie = pd.Series(serie).dropna()
#     stat, p, *_ = adfuller(serie, autolag="AIC")
#     veredicto = "estacionaria" if p < 0.05 else "NO estacionaria"
#     print(f"ADF · {nombre}: estad = {stat:.3f}, p = {p:.4f} → {veredicto}")
#
#     fig, ax = plt.subplot_mosaic(
#         [["serie", "serie"], ["acf", "pacf"]], figsize=(12, 6)
#     )
#     serie.plot(ax=ax["serie"], lw=0.5)
#     ax["serie"].axhline(serie.mean(), color="k", lw=0.5, alpha=0.5)
#     ax["serie"].set_title(f"Serie — {nombre}")
#     plot_acf(serie,  lags=lags, ax=ax["acf"]);  ax["acf"].set_title("ACF")
#     plot_pacf(serie, lags=lags, ax=ax["pacf"], method="ywm")
#     ax["pacf"].set_title("PACF")
#     plt.tight_layout(); plt.show()
#
#
# def diagnostico(modelo, nombre, lags=20):
#     """Validación: residuales + ACF + PACF + Ljung-Box."""
#     p, d, q = modelo.model.order
#     resid = modelo.resid.iloc[max(p, q):]
#
#     fig, axes = plt.subplots(1, 3, figsize=(15, 3.2))
#     axes[0].plot(resid.values, lw=0.6); axes[0].axhline(0, color="k", lw=0.5)
#     axes[0].set_title(f"Residuales — {nombre}")
#     plot_acf(resid,  lags=lags, ax=axes[1]); axes[1].set_title("ACF resid.")
#     plot_pacf(resid, lags=lags, ax=axes[2], method="ywm")
#     axes[2].set_title("PACF resid.")
#     plt.tight_layout(); plt.show()
#
#     lb = acorr_ljungbox(resid, lags=[10, 20], return_df=True)
#     print(f"Ljung-Box · {nombre}\n{lb.round(4)}")
#     return lb
# ```

# %% [markdown]
# ## 11 · Mini-mapa de decisión
#
# ```
#                      ┌─────────────────────────────┐
#                      │   Serie temporal y_t        │
#                      └──────────────┬──────────────┘
#                                     ▼
#                  ┌──────────────────────────────────────┐
#                  │ ¿Tendencia / raíz unitaria? (ADF)    │
#                  └──────┬──────────────────────┬────────┘
#                     sí  ▼                    no ▼
#              Diferenciar Δy_t          ┌──────────────────────────┐
#              → ARIMA(p,1,q)            │ ¿Estacionalidad? (ACF,   │
#                     │                  │   climatología, picos s) │
#                     │                  └──────┬────────────┬──────┘
#                     │                     sí  ▼          no ▼
#                     │             Restar climatología   ACF/PACF →
#                     │             o SARIMA(p,d,q)        orden (p,q)
#                     │             (P,D,Q)_s                   │
#                     ▼                     ▼                   ▼
#                     └─────────────────────┴───────────────────┘
#                                     │
#                                     ▼
#                       Estimar ARIMA(p,d,q).fit()
#                                     │
#                                     ▼
#                   ┌─────────────────────────────────┐
#                   │ Ljung-Box sobre residuales      │
#                   └────────┬───────────────┬────────┘
#                       rechaza ▼          ok ▼
#                       Subir orden        ¿AIC sigue bajando con
#                       (vuelve a fit)      coef. significativos?
#                                                │       │
#                                            sí  ▼     no ▼
#                                          Iterar     PARAR
#                                                       │
#                                                       ▼
#                                       Backtest vs baseline naive
#                                       (persistencia, climatología)
# ```

# %% [markdown]
# ## 12 · Lo que aprendimos en La Ventosa (en una frase cada uno)
#
# 1. **ADF puede confundirse con estacionalidad fuerte** — siempre verifica con ACF
#    y climatología.
# 2. **Restar climatología por hora simplifica el modelo**, pero deja un ciclo
#    residual día-a-día que sobrevive en ACF cerca de lag 24.
# 3. **$\phi \approx 0.97$ → vida media $\approx 23$ h** — casi raíz unitaria;
#    considera ARIMA$(p,1,q)$ como alternativa.
# 4. **ARMA(2,1) gana en AIC** pero sus raíces están al borde de la estacionariedad;
#    señal de potencial sobreajuste o necesidad de diferenciar.
# 5. **Backtest contra persistencia ingenua** puede mostrar que el modelo "más
#    estadísticamente correcto" pierde en MAE — Box-Jenkins es necesario, no suficiente.
# 6. **Sin desestacionalizar, ARMA es plano** — incapaz de reproducir el ciclo
#    diario, MAE empeora. Es la frontera operativa de la familia ARMA.
#
# **Próximo paso natural:** SARIMA y SARIMAX (con exógenas como presión, gradiente
# térmico, índices climáticos) — motivado por necesidad observada, no por currículum.

# %% [markdown]
# ## Lecturas
#
# - Box, Jenkins, Reinsel, Ljung (2015). *Time Series Analysis: Forecasting and
#   Control* (5th ed.). Caps. 3–4.
# - Cryer & Chan (2008). *Time Series Analysis: With Applications in R*. Caps. 4–6.
# - Hyndman & Athanasopoulos. *Forecasting: Principles and Practice* —
#   https://otexts.com/fpp3/ (cap. 9).
