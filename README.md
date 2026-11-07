# Airwave Architects - Airtel Sync 2026 (Network Track)

## What this project does

We built a small simulation of a mobile backhaul network and tried to answer one question: can we see a congested link coming before it happens, and move traffic away from it in time?

The network has 12 nodes and 22 links (8 towers, 2 hubs, a core and a satellite node) using fiber, microwave and satellite links. We generate 21 days of fake tower traffic at 15 minute intervals, train a LightGBM model to forecast total demand 30 minutes ahead, and then use that forecast in two ways:

1. Bottleneck prediction. We push the forecast traffic through the network and mark any link that goes over 80% utilization. Then we check these predictions against what really happened.
2. Rerouting. We compare normal shortest path routing with a congestion aware routing method on the same forecast traffic.

None of the data is real Airtel data. All traffic, capacities, latencies and costs are made up for the demo.

## How to run

Install the packages:

```
pip install -r requirements.txt
```

Run everything (data generation, model, routing, comparison, charts and checks):

```
python main.py
```

It stops at the first step that fails. After it finishes you can open the dashboard:

```
streamlit run dashboard/app.py
```

## Files it produces

- `data/raw/traffic.csv` - the synthetic tower traffic
- `data/processed/forecast_dataset.csv` - features for the forecasting model
- `data/processed/forecast_predictions.csv` - actual vs LightGBM vs persistence
- `data/processed/forecast_adjusted_traffic.csv` - tower traffic built from the forecast
- `data/processed/baseline_links.csv` - baseline routing on the actual traffic
- `data/processed/test_baseline_links.csv` - baseline routing on the forecast traffic
- `data/processed/optimized_links.csv` - congestion aware routing on the forecast traffic
- `data/processed/forecast_metrics.json`, `comparison_metrics.json`, `bottleneck_prediction_metrics.json` - the numbers
- `data/processed/bottleneck_episodes.csv` - each congestion episode and whether we flagged it early
- `results/` - charts
- `models/traffic_forecaster.pkl` - the trained model

## How the bottleneck prediction is scored

`predict_bottlenecks.py` compares two runs of the same baseline routing. One uses the forecast traffic (built from a forecast made 30 minutes earlier) and the other uses the actual traffic. A link counts as a bottleneck when it is active and above 80% utilization. We then count true positives, false positives and false negatives for every link and 15 minute interval.

We also look at congestion episodes, meaning a run of back to back congested intervals on one link. An episode counts as caught if we flagged that link at any point during it, and caught at onset if we flagged it in the very first interval.

## Results from our run

These are saved in the json files and get recalculated every time you run `main.py`.

Forecast model (384 test samples):

- MAE 0.0761 Gbps, RMSE 0.1419 Gbps, R2 0.9979
- A simple "same as now" baseline has MAE 0.3399 Gbps, so the model is 77.63% better

Bottleneck prediction (30 minutes ahead, 80% threshold, 8,448 link intervals):

- 68 real bottleneck records, 68 predicted
- 66 correct, 2 false alarms, 2 missed
- Precision, recall and F1 are all 97.1%
- 4 congestion episodes in total, all 4 flagged at least once, 3 of 4 flagged at the start

Routing comparison on the same forecast traffic:

- Baseline: 68 congested link records (0.82% of active records)
- Optimized: 30 congested link records (0.36%)
- That is a 56.05% drop in congestion
- Peak utilization goes from 982.78% to 97.50%, a 90.08% drop
- No unrouted demand in either case

These only describe this synthetic test. They are not results from a real network.

## Limitations

- All data is synthetic, including the festival and outage scenarios.
- The model forecasts total network demand, not individual towers or links. To split it across towers we use each tower's share at the moment the forecast was made. Link level bottlenecks come from routing that forecast, the model does not predict them directly.
- The festival (days 12 and 18, 17:00 to 22:00, extra demand at T3, T4 and T5) is treated as a known, announced event.
- The fiber outage on 20 Oct is given to the system as a known state. The model does not predict it.
- In the optimized run, two extra 3 Gbps hub to core microwave backup links are switched on during the outage. The baseline does not get these. So part of the peak utilization improvement comes from the extra capacity and not only from the routing logic.
- The 30 minute lead time is built into how the test is set up, since the forecast traffic for each time comes from a forecast made 30 minutes before.
- The routing is a greedy heuristic. It is not an optimal solver and it does not enforce hard capacity limits.
- Utilization above 100% means the link is overloaded in the model, not that it carried that much traffic.
- The test window only has 4 congestion episodes, so the bottleneck numbers are a rough indication and not something statistically solid.
