import pandas as pd
import xgboost as xgb
import joblib
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Load credentials
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("meta_credentials.json", scope)
client = gspread.authorize(creds)

# Load training data
sheet = client.open("meta_model_training").sheet1
rows = sheet.get_all_values()
headers = rows[0]
data = rows[1:]

df = pd.DataFrame(data, columns=headers)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(inplace=True)

# Train model
X = df.drop(columns=["final_outcome"])
y = df["final_outcome"]
model = xgb.XGBClassifier(eval_metric="logloss")
model.fit(X, y)

# Save model
joblib.dump(model, "meta_model.pkl")
print("✅ Meta model trained and saved as meta_model.pkl")
