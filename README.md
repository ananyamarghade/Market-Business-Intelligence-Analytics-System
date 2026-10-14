# Market and Business Intelligence Analytics System

## Run

```
pip install -r requirements.txt
python src/generate_data.py
python src/analysis.py
```

Open `dashboard/index.html` in a browser.

## Structure

```
data/monthly_sales.csv    monthly revenue by category and region with drivers
src/generate_data.py      builds the dataset and dashboard/data.js
src/analysis.py           decomposition, forecast, backtest, OLS, ACF, figures
dashboard/index.html      interactive dashboard
reports/                  summary.json, forecast.csv, figures
```
