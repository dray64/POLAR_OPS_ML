def assess_dataset(df):
    counts = df["high_risk_label"].value_counts()
    positives = int(counts.get(1, 0))
    negatives = int(counts.get(0, 0))
    return {
        "status": "NOT_READY" if positives < 10 else "READY_FOR_TRAINING",
        "rows": int(len(df)),
        "positive_rows": positives,
        "negative_rows": negatives,
        "reason": (
            "Too few positive examples for reliable binary classification."
            if positives < 10 else ""
        ),
    }
