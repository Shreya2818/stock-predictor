def run_prediction():
    import yfinance as yf
    import numpy as np
    import datetime
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import load_model
    from datetime import datetime
    import pytz

    def is_market_open_india():
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)
        weekday = now.weekday()
        hour, minute = now.hour, now.minute
        return (weekday < 5) and (
            (hour == 9 and minute >= 15) or (9 < hour < 15) or (hour == 15 and minute < 30)
        )

    if not is_market_open_india():
        print("❌ Market is closed.")
        return None, None

    df = yf.download("RPOWER.NS", interval="1m", period="1d")
    df.dropna(inplace=True)
    df = df.tail(60)

    if df.shape[0] < 60:
        print("⏳ Not enough data.")
        return None, None

    close = df[['Close']]
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(close)
    X_live = scaled_close.reshape(1, 60, 1)

    model = load_model("rpower_lstm_model.h5")
    pred_scaled = model.predict(X_live)
    pred_price = scaler.inverse_transform(pred_scaled)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    price = round(pred_price[0][0], 2)

    print(f"🕒 Current Time: {now}")
    print(f"🔮 Predicted RPOWER Price after 5 min: ₹{price}")

    return now, price

# ✅ Run prediction if this file is executed directly
if __name__ == "__main__":
    run_prediction()
