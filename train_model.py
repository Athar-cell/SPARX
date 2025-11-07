
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

n = 1000
np.random.seed(123)
training_duration = np.random.normal(60, 25, n).clip(10, 300)
rpe = np.random.randint(2, 10, n)
sleep_hours = np.random.normal(7, 1.8, n).clip(3, 10)
prev_injuries = np.random.binomial(1, 0.18, n)
sprint_distance = np.random.normal(400, 250, n).clip(0, 3000)
avg_hr = np.random.normal(140, 18, n).clip(80, 210)

risk_score = 0.02*training_duration + 0.35*rpe - 0.45*sleep_hours + 1.2*prev_injuries + 0.0012*sprint_distance + 0.012*(avg_hr-120)
prob = 1 / (1 + np.exp(- (risk_score - 6.5)))
injury = (np.random.rand(n) < prob).astype(int)

df = pd.DataFrame({
    "training_duration": training_duration,
    "rpe": rpe,
    "sleep_hours": sleep_hours,
    "prev_injuries": prev_injuries,
    "sprint_distance": sprint_distance,
    "avg_hr": avg_hr,
    "injury": injury
})
df.to_csv("sample_data.csv", index=False)

X = df.drop(columns=["injury"])
y = df["injury"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

joblib.dump(model, "model.joblib")
print("Trained model saved as model.joblib and sample_data.csv generated.")
