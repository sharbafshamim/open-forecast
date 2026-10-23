#!/bin/bash
# Reproduce every experiment in the paper. Safe to re-run: each step skips runs
# whose metrics file already exists (price runs also cache per-model forecasts,
# rolling runs cache per-window results).
#
#   pip install -e ".[all]"
#   python scripts/download_data.py        # + place ISO-NE / Tetouan files (data/README.md)
#   bash scripts/reproduce.sh
#
# Foundation models download weights from the Hugging Face Hub on first use.
# Device: OPEN_FORECAST_DEVICE=mps (Apple silicon) | cuda | cpu
set -e
cd "$(dirname "$0")/.."
export OPEN_FORECAST_DEVICE=${OPEN_FORECAST_DEVICE:-cpu}
MODES="weather_time weather_time_history weather_time_history_weekly"
NAIVE="persistence seasonal_naive"
TREES="random_forest extra_trees hist_gradient_boosting"
DEEP="lstm cross_attention_transformer patchtst"
FM="chronos_bolt_small chronos_bolt_base chronos2"

run_load () {  # config output models...
  local cfg=$1 out=$2; shift 2
  [ -f "$out/metrics.csv" ] && { echo "skip $out"; return; }
  python -u -m open_forecast.cli run --config "$cfg" --models "$@" --modes $MODES --output "$out"
}

# 1. Load forecasting, single chronological split (all model families share origins)
for ds in isone tetouan_v2; do
  run_load configs/$ds.yaml results/${ds}_naive_trees $NAIVE $TREES
  run_load configs/$ds.yaml results/${ds}_deep        $NAIVE $DEEP
  run_load configs/$ds.yaml results/${ds}_foundation  $NAIVE $FM
done

# 2. Rolling-origin evaluation on Tetouan (11 x 28-day windows)
python -u -m open_forecast.cli rolling --config configs/tetouan_rolling.yaml \
  --models $NAIVE $TREES $DEEP chronos_bolt_small chronos2 --modes $MODES \
  --output results/tetouan_rolling

# 3. Day-ahead price forecasting (ISO-NE + official EPF markets)
for cfg in isone_price epf_pjm epf_np epf_de epf_be epf_fr; do
  out=results/price_${cfg}
  [ -f "$out/price_metrics.csv" ] && { echo "skip $out"; continue; }
  market=$(echo "${cfg#epf_}" | tr a-z A-Z)
  if [ "$cfg" != isone_price ] && [ ! -f "data/raw/epf/${market}.csv" ]; then
    echo "missing data/raw/epf/${market}.csv (run scripts/download_data.py)"; continue
  fi
  python -u -m open_forecast.cli price --config configs/${cfg}.yaml --output $out \
    --models naive lear hist_gradient_boosting extra_trees $FM
done

# 4. Significance tests (load) and paper tables / figures
for d in results/*_naive_trees results/*_deep results/*_foundation; do
  [ -f "$d/predictions.csv" ] && python analysis/dm_tests.py "$d/predictions.csv" seasonal_naive || true
done
python analysis/summarize.py results
python analysis/eda_figures.py --data data/raw --out results/eda
