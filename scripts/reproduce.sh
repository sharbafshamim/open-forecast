#!/bin/bash
# Reproduce every experiment in the paper, then rebuild its tables and figures.
# Safe to re-run: finished runs are skipped (load runs per window, price runs per
# model through a configuration-keyed forecast cache).
#
#   pip install -e ".[all]"
#   python scripts/download_data.py             # EPF markets; place ISO-NE / Tetouan files (data/README.md)
#   python scripts/fetch_weather_forecasts.py    # archived day-ahead weather forecasts (optional)
#   bash scripts/reproduce.sh
#
# Foundation models download weights from the Hugging Face Hub on first use.
# Device: OPEN_FORECAST_DEVICE=mps (Apple silicon) | cuda | cpu
# The folder names below are exactly the ones analysis/paper_tables.py reads.
set -e
cd "$(dirname "$0")/.."
export OPEN_FORECAST_DEVICE=${OPEN_FORECAST_DEVICE:-cpu}
MODES="weather_time weather_time_history weather_time_history_weekly"
NAIVE="persistence seasonal_naive"
TREES="random_forest extra_trees hist_gradient_boosting"
CLI="python -u -m open_forecast.cli"

# 1. ISO-NE load: 13 rolling 28-day windows over 2025, every model refit per window
$CLI rolling --config configs/isone_rolling.yaml --models $NAIVE $TREES --modes $MODES \
  --output results/isone_rolling_trees
$CLI rolling --config configs/isone_rolling.yaml --models chronos_bolt_small chronos2 \
  --modes weather_time_history_weekly --output results/isone_rolling_foundation
for m in lstm cross_attention_transformer patchtst; do
  $CLI rolling --config configs/isone_rolling.yaml --models $m --modes $MODES \
    --output results/isone_rolling_$m
done

# 2. ISO-NE load with archived day-ahead weather forecasts at test time
FC=data/raw/isone/weather_forecasts_previous_day2.csv
if [ -f "$FC" ]; then
  $CLI rolling --config configs/isone_rolling.yaml --weather-forecast $FC \
    --models persistence random_forest hist_gradient_boosting chronos2 --modes $MODES \
    --output results/isone_rolling_forecastweather_trees
  $CLI rolling --config configs/isone_rolling.yaml --weather-forecast $FC --models lstm --modes $MODES \
    --output results/isone_rolling_forecastweather_lstm
fi

# 3. Tetouan: single split and 11 rolling windows
$CLI run --config configs/tetouan_v2.yaml --models $NAIVE $TREES --modes $MODES --output results/tetouan_v2_trees
$CLI run --config configs/tetouan_v2.yaml --models $NAIVE lstm cross_attention_transformer patchtst \
  --modes $MODES --output results/tetouan_v2_deep
$CLI run --config configs/tetouan_v2.yaml --models $NAIVE chronos_bolt_small chronos_bolt_base chronos2 \
  --modes $MODES --output results/tetouan_v2_foundation
$CLI rolling --config configs/tetouan_rolling.yaml --models $NAIVE $TREES --modes $MODES \
  --output results/tetouan_rolling_trees
$CLI rolling --config configs/tetouan_rolling.yaml --models lstm cross_attention_transformer patchtst \
  --modes $MODES --output results/tetouan_rolling_deep
$CLI rolling --config configs/tetouan_rolling.yaml --models chronos_bolt_small chronos2 \
  --modes weather_time_history_weekly --output results/tetouan_rolling_foundation

# 4. Day-ahead price: ISO-NE (bidding-gate inputs) and the five EPF markets
for cfg in isone_price epf_pjm epf_np epf_de epf_be epf_fr; do
  market=$(echo "${cfg#epf_}" | tr a-z A-Z)
  if [ "$cfg" != isone_price ] && [ ! -f "data/raw/epf/${market}.csv" ]; then
    echo "missing data/raw/epf/${market}.csv (run scripts/download_data.py)"; continue
  fi
  $CLI price --config configs/${cfg}.yaml --output results/price_${cfg} \
    --models naive lear hist_gradient_boosting extra_trees chronos_bolt_small chronos_bolt_base chronos2
done

# 5. Forecast combination Chronos-2 + LEAR (equal weights) on every price market
python analysis/price_ensemble.py results/price_isone_price
for m in pjm np de be fr; do
  M=$(echo $m | tr a-z A-Z)
  [ -d results/price_epf_$m ] && python analysis/price_ensemble.py results/price_epf_$m \
    data/external/epftoolbox_forecasts/Forecasts_${M}_DNN_LEAR_ensembles.csv
done

# 6. Tables, figures and significance tests of the paper -> results/paper/
python analysis/paper_tables.py results
python analysis/eda_figures.py --data data/raw --out results/eda
