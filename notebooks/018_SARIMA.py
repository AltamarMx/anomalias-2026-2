import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt

    return (pd,)


@app.cell
def _(pd):
    f = "data/ClimaLab_2023-05-31_2025-06-20.parquet"
    tdb = pd.read_parquet(f,columns=["tdb"])
    tdb.plot()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
