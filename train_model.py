import pandas as pd
import glob
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("📦 Loading all transaction files...")
all_files = sorted(glob.glob("dataset/*.pkl"))

dfs = [pd.read_pickle(f) for f in all_files]
df = pd.concat(dfs, ignore_index=True)
print(f"✅ Loaded {len(df)} total transactions.")
print(df.head())

df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
df['TX_DAY'] = df['TX_DATETIME'].dt.day
df['TX_HOUR'] = df['TX_DATETIME'].dt.hour

features = ['TX_AMOUNT', 'TX_DAY', 'TX_HOUR']
X = df[features]
y = df['TX_FRAUD']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("🤖 Training RandomForest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n📊 Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(model, "fraud_model.pkl")
joblib.dump(features, "model_columns.pkl")
print("✅ Model trained and saved successfully!")
