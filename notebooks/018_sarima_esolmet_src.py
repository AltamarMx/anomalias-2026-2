import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


    return SARIMAX, mo, np, pd, plot_acf, plot_pacf, plt


@app.cell
def _(pd):
    f = 'data/ClimaLab_2023-05-31_2025-06-20.parquet'
    tdb_all = pd.read_parquet(f)['tdb']
    tdb_h = tdb_all.resample('h').mean().dropna()
    tdb_h.plot(figsize=(10, 3), title='tdb horaria — ESOLMET (todo el periodo)')

    return (tdb_h,)


@app.cell
def _(tdb_h):
    tdb = tdb_h.loc['2025-05-01':'2025-06-20']
    y = tdb.astype(float).asfreq('h').interpolate('time')
    gaps = int(tdb.asfreq('h').isna().sum())
    tdb.plot(figsize=(10, 3), title=f'tdb horaria — ventana de trabajo ({len(y)} h, {gaps} huecos interpolados)')

    return (y,)


@app.cell(hide_code=True)
def _(mo, y):
    horizon = 240
    y_train = y.iloc[:-horizon]
    y_test = y.iloc[-horizon:]
    mo.md(
        f"Train: **{len(y_train)}** h ({y_train.index[0]} → {y_train.index[-1]})  \n"
        f"Test:  **{len(y_test)}** h ({y_test.index[0]} → {y_test.index[-1]})"
    )

    return y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    order_form = (
        mo.md(
            """
            **Non-seasonal**  p = {p}, d = {d}, q = {q}

            **Seasonal**      P = {P}, D = {D}, Q = {Q}, s = {s}
            """
        )
        .batch(
            p=mo.ui.slider(0, 5, value=2, label="p"),
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
def _(SARIMAX, mo, np, order_form, plot_acf, plot_pacf, plt, y_test, y_train):
    cfg = order_form.value
    mo.stop(cfg is None, mo.md("Pick orders above and click **Fit SARIMA**."))

    order = (cfg["p"], cfg["d"], cfg["q"])
    seasonal_order = (cfg["P"], cfg["D"], cfg["Q"], cfg["s"])

    with mo.status.spinner("Fitting SARIMA..."):
        res = SARIMAX(
            y_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

    fc = res.get_forecast(steps=len(y_test))
    y_hat = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)

    burn = max(seasonal_order[3], 24)
    resid = res.resid.iloc[burn:]

    fig, axes = plt.subplots(4, 1, figsize=(11, 12))

    ax = axes[0]
    y_train.iloc[-7 * 24:].plot(ax=ax, label="train (last 7 d)")
    y_test.plot(ax=ax, label="test", color="black")
    y_hat.plot(ax=ax, label="forecast", color="red")
    ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color="red", alpha=0.15)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Forecast vs holdout")
    ax.set_ylabel("tdb (°C)")

    ax = axes[1]
    resid.plot(ax=ax)
    ax.set_title("Residuals (after burn-in)")
    ax.axhline(0, color="k", lw=0.5)

    plot_acf(resid, ax=axes[2], lags=48)
    plot_pacf(resid, ax=axes[3], lags=48, method="ywm")

    plt.tight_layout()

    rmse = float(np.sqrt(((y_test - y_hat) ** 2).mean()))
    mae = float((y_test - y_hat).abs().mean())

    mo.vstack([
        mo.md(
            f"**SARIMA{order} x {seasonal_order}** &nbsp;·&nbsp; "
            f"AIC = `{res.aic:.2f}` &nbsp;·&nbsp; BIC = `{res.bic:.2f}` &nbsp;·&nbsp; "
            f"RMSE(test) = `{rmse:.2f} °C` &nbsp;·&nbsp; MAE(test) = `{mae:.2f} °C`"
        ),
        fig,
        mo.accordion({"Full summary": mo.md(f"```\n{res.summary().as_text()}\n```")}),
    ])

    return


if __name__ == "__main__":
    app.run()
