import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
HOLDOUT = 6
HORIZON = 6


def load():
    df = pd.read_csv(ROOT / "data" / "monthly_sales.csv", parse_dates=["month"])
    total = df.groupby("month")["revenue"].sum()
    drivers = df.drop_duplicates("month").set_index("month")[["promo", "price_index", "temperature", "traffic"]]
    return df, total, drivers


def fit_seasonal(y, n):
    ols = sm.OLS(np.log(y.iloc[:n].to_numpy()), sm.add_constant(np.arange(n))).fit()
    trend = lambda k: np.exp(ols.params[0] + ols.params[1] * k)
    months = y.index.month
    idx = {m: np.mean([y.iloc[i] / trend(i) for i in range(n) if months[i] == m]) for m in range(1, 13)}
    predict = lambda k, month: trend(k) * idx[month]
    sigma = np.std([np.log(y.iloc[i] / predict(i, months[i])) for i in range(n)], ddof=1)
    return predict, sigma


def backtest(y):
    n = len(y) - HOLDOUT
    predict, _ = fit_seasonal(y, n)
    actual = y.iloc[n:].to_numpy()
    model = np.array([predict(n + i, y.index[n + i].month) for i in range(HOLDOUT)])
    naive = y.shift(12).iloc[n:].to_numpy()
    score = lambda p: {"mape": float(np.mean(np.abs(p - actual) / actual)),
                       "rmse": float(np.sqrt(np.mean((p - actual) ** 2)))}
    return {"seasonal_naive": score(naive), "seasonal_trend": score(model)}


def forecast(y):
    n = len(y)
    predict, sigma = fit_seasonal(y, n)
    future = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=HORIZON, freq="MS")
    point = np.array([predict(n + h, future[h].month) for h in range(HORIZON)])
    width = np.array([sigma * (1 + .04 * h) * 1.2816 for h in range(HORIZON)])
    return pd.DataFrame({"forecast": point, "lower": point * np.exp(-width), "upper": point * np.exp(width)}, index=future)


def drivers_model(y, drivers):
    X = drivers[["promo", "price_index", "temperature"]].copy()
    X["trend"] = np.arange(len(y))
    return sm.OLS(np.log(y.to_numpy()), sm.add_constant(X)).fit()


def category_shift(df):
    pivot = df.pivot_table(index="month", columns="category", values="revenue", aggfunc="sum")
    return (pivot.iloc[-12:].sum() / pivot.iloc[-24:-12].sum() - 1).sort_values(ascending=False)


def plots(y, drivers, decomposition, fc):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(y.index, y, color="#1F77B4", label="Observed")
    ax.plot(fc.index, fc["forecast"], color="#E8710A", label="Forecast")
    ax.fill_between(fc.index, fc["lower"], fc["upper"], color="#E8710A", alpha=.2, label="80% interval")
    ax.set_ylabel("Monthly revenue")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "forecast.png", dpi=150)
    plt.close(fig)

    fig = decomposition.plot()
    fig.set_size_inches(10, 7)
    fig.tight_layout()
    fig.savefig(OUT / "decomposition.png", dpi=150)
    plt.close(fig)

    x = np.arange(len(y))
    detrended = y.to_numpy() - np.polyval(np.polyfit(x, y.to_numpy(), 1), x)
    values = acf(detrended, nlags=18)[1:]
    bound = 1.96 / np.sqrt(len(y))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(range(1, 19), values, color="#1F77B4")
    ax.axhline(bound, color="#D62728", ls="--")
    ax.axhline(-bound, color="#D62728", ls="--")
    ax.set_xlabel("Lag (months)")
    fig.tight_layout()
    fig.savefig(OUT / "acf.png", dpi=150)
    plt.close(fig)

    corr = pd.concat([y.rename("revenue"), drivers], axis=1).corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(OUT / "correlation.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    df, y, drivers = load()
    decomposition = seasonal_decompose(y, model="additive", period=12)
    resid = decomposition.resid.dropna()
    z = (resid - resid.mean()) / resid.std()
    fc = forecast(y)
    model = drivers_model(y, drivers)
    strength = max(0.0, 1 - resid.var() / (decomposition.seasonal + decomposition.resid).dropna().var())

    summary = {
        "revenue_last_12m": float(y.iloc[-12:].sum()),
        "yoy_growth": float(y.iloc[-12:].sum() / y.iloc[-24:-12].sum() - 1),
        "seasonal_strength": float(strength),
        "anomaly_months": [d.strftime("%Y-%m") for d in z[z.abs() > 2].index],
        "backtest": backtest(y),
        "driver_model": {
            "r_squared": float(model.rsquared),
            "coefficients": model.params.round(5).to_dict(),
            "t_values": model.tvalues.round(2).to_dict(),
        },
        "category_yoy": category_shift(df).round(4).to_dict(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    fc.round(2).to_csv(OUT / "forecast.csv")
    plots(y, drivers, decomposition, fc)
    print(json.dumps(summary["backtest"], indent=2))


if __name__ == "__main__":
    main()
