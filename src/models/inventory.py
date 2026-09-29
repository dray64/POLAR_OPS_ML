import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train(df, features):
    x = df.sort_values("date").copy()
    split = int(len(x) * 0.80)
    train_df, test_df = x.iloc[:split], x.iloc[split:]

    model = XGBRegressor(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(train_df[features], train_df["next_day_demand"])

    pred = model.predict(test_df[features])
    y = test_df["next_day_demand"]

    metrics = {
        "mae": mean_absolute_error(y, pred),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": r2_score(y, pred),
        "test_rows": len(test_df),
    }
    return model, metrics, test_df, y, pred
