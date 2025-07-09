import schedule
import time
import requests
from live_predict import run_prediction

# ✅ Airtable Config
API_KEY = "YOUR_PAT_TOKEN"
BASE_ID = "YOUR_BASE_ID"
TABLE_NAME = "Live Predictions"

def send_to_airtable(timestamp, price):
    url = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME}"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "fields": {
            "Time": timestamp,
            "Predicted Price": price
        }
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code in [200, 201]:
        print(f"✅ Logged to Airtable: {timestamp} — ₹{price}")
    else:
        print("❌ Airtable Error:", response.text)

def job():
    timestamp, price = run_prediction()
    if timestamp and price:
        send_to_airtable(timestamp, price)
    else:
        print("⏭️ Skipped: No prediction (market closed or insufficient data)")

# 🔁 Schedule the job every 5 minutes
schedule.every(5).minutes.do(job)
print("🟢 Automation started: Predicting every 5 minutes...")

# 🕒 Run continuously
while True:
    schedule.run_pending()
    time.sleep(1)
