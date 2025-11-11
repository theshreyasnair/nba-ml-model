import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

def train_and_evaluate(model, df, features, target="home_win", scale=False):
    df = df.dropna(subset=features + [target])

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n=== {model.__class__.__name__} ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"ROC AUC: {auc:.3f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return model

if __name__ == "__main__":
    df = pd.read_csv("../data/processed/2023_24_features.csv")

    features = [
        "avg_pts_lastN_HOME",
        "avg_pts_lastN_AWAY",
        "rest_days_HOME",
        "rest_days_AWAY",
        "rest_diff",
        "elo_HOME",
        "elo_AWAY"
    ]
    lr = LogisticRegression(max_iter=1000)
    train_and_evaluate(lr, df, features, scale=True)

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42
    )
    train_and_evaluate(rf, df, features, scale=False)
