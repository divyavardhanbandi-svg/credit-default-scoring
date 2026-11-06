# Credit Default Scoring

Predicts the probability that a borrower will default and converts it into a scorecard-style credit score with risk bands.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# Built-in synthetic data (no download needed)
python credit_default_scoring.py

# Real dataset, e.g. UCI "Default of Credit Card Clients"
python credit_default_scoring.py --data UCI_Credit_Card.csv --target default.payment.next.month
```

## What it does

1. Loads data (synthetic by default, or your CSV)
2. Preprocesses: median/mode imputation, scaling, one-hot encoding
3. Trains Logistic Regression and Gradient Boosting
4. Evaluates with AUC, Gini, KS, confusion matrix and classification report
5. Converts default probability to a credit score (base 600 at 50:1 odds, 20 points to double the odds)
6. Assigns risk bands and saves results

## Outputs (in `outputs/`)

- `best_model.joblib` - trained pipeline
- `roc_curve.png` - ROC curves for both models
- `scored_test_set.csv` - test set with probability, score and risk band
