# Simple Flask Leaderboard App

This app lets you upload student prediction files and automatically ranks teams.

## Expected submission format
Each team uploads a CSV file with these columns:

- `customer_id`
- `churn_prediction`

Example:

```csv
customer_id,churn_prediction
CUST100960,0
CUST100963,1
```

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Then open the local URL shown in the terminal.

## Important
Place your hidden ground-truth file in the app folder with this exact name:

`ml_bi_hackathon_test.csv`

The file must contain:
- `customer_id`
- `churned`

## Ranking logic
Teams are ranked by:
1. `f1`
2. `accuracy`

If the same team uploads again, the latest submission replaces the previous one.
