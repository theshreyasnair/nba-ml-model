import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

def train_baseline_model():
    scaler = StandardScaler()

    df = pd.read_csv("../data/processed/2023_24_features.csv")
    df = df.dropna(subset = ["avg_pts_lastN_HOME", "avg_pts_lastN_AWAY", "home_win"])

    X = df[["avg_pts_lastN_HOME", "avg_pts_lastN_AWAY"]]
    y = df["home_win"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = False)

    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("ROC AUC: ", roc_auc_score(y_test, y_proba))
    print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))

    return model

if __name__ == "__main__":
    train_baseline_model()

'''
loads engineered data
drops nan values/missing features
uses 80 for training and 20 for testing
trains a logistic regression model

accuracy -> accuracy
roc auc -> how well it ranks games by confidence
confusion matrix -> breakdown of t/f +/-
'''