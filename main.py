import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime
import logging
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization

import joblib
import pmdarima as pm  # For auto_arima

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


# --- CONFIG ---
# --- CONFIG ---
CSV_FILE_PATH = "/kaggle/input/crude-oil-price-prediction/Crude oil.csv"  # ✅ Exact filename
MODEL_PATH = "/kaggle/working/lstm_model.h5"
SCALER_PATH = "/kaggle/working/scaler.save"
ARIMA_PATH = "/kaggle/working/arima_model.pkl"
PREDICTIONS_PATH = "/kaggle/working/predictions.csv"
LOOK_BACK = 90
ADMIN_TOKEN = "supersecret"

# --- DATA LOADING ---
def load_data(url):
    try:
        logger.info(f"Loading data from {url}")
        df = pd.read_csv(url)
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col])
                df = df.sort_values(col)
                break
        # --- Outlier capping for 'Volume' column ---
        if 'Volume' in df.columns:
            lower_cap = df['Volume'].quantile(0.05)
            upper_cap = df['Volume'].quantile(0.95)
            print(f"Lower cap (5th percentile): {lower_cap}")
            print(f"Upper cap (95th percentile): {upper_cap}")
            df['Volume'] = np.where(df['Volume'] > upper_cap, upper_cap,
                                    np.where(df['Volume'] < lower_cap, lower_cap, df['Volume']))
            if df['Volume'].isna().any():
                median_vol = np.nanmedian(df['Volume'])
                logger.warning("NaNs found in 'Volume' after capping. Filling with median: {}".format(median_vol))
                df['Volume'] = df['Volume'].fillna(median_vol)
        else:
            print("No 'Volume' column found in the DataFrame.")
        if 'Close/Last' in df.columns:
            if df['Close/Last'].isna().any():
                median_close = np.nanmedian(df['Close/Last'])
                logger.warning("NaNs found in 'Close/Last'. Filling with median: {}".format(median_close))
                df['Close/Last'] = df['Close/Last'].fillna(median_close)
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def get_series(df):
    if 'Close' in df.columns:
        return df['Close'].values
    elif 'Close/Last' in df.columns:
        return df['Close/Last'].values
    else:
        logger.error("No 'Close' or 'Close/Last' column found.")
        raise ValueError("No 'Close' or 'Close/Last' column found.")

def preprocess(train_features, test_features):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_features)
    scaled_test = scaler.transform(test_features)
    joblib.dump(scaler, SCALER_PATH)
    return scaled_train, scaled_test, scaler

def create_sequences(data, look_back=LOOK_BACK):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, :])  # Use all features
        y.append(data[i, 0])  # Predict residual (first feature)
    X, y = np.array(X), np.array(y)
    if len(X) == 0:
        return np.empty((0, look_back, data.shape[1])), np.empty((0,))
    return X, y

def build_lstm(input_shape):
    model = Sequential([
        LSTM(8, input_shape=input_shape, kernel_regularizer=regularizers.l2(0.1)),
        Dropout(0.7),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def remove_nan_rows(*arrays):
    """Remove rows where any array has a NaN in that row (works for 1D, 2D, or 3D arrays)."""
    mask = np.ones(arrays[0].shape[0], dtype=bool)
    for arr in arrays:
        if arr.ndim == 1:
            mask &= ~np.isnan(arr)
        elif arr.ndim == 2:
            mask &= ~np.isnan(arr).any(axis=1)
        elif arr.ndim == 3:
            mask &= ~np.isnan(arr).reshape((arr.shape[0], -1)).any(axis=1)
        else:
            raise ValueError("Unsupported array dimension in remove_nan_rows")
    return [arr[mask] for arr in arrays]

def train_and_predict(df):
    try:
        # Use only 'Close/Last' and 'Volume'
        if len(df) < LOOK_BACK * 2:
            logger.error(f"Not enough data for training (need at least {LOOK_BACK*2} rows)")
            return
        close_series = df['Close/Last'].values
        volume_series = df['Volume'].values
        split_idx = int(0.8 * len(df))
        train_close, test_close = close_series[:split_idx], close_series[split_idx:]
        train_volume, test_volume = volume_series[:split_idx], volume_series[split_idx:]

        # --- ARIMA (auto-tuned) on Close/Last ---
        logger.info("Auto-tuning ARIMA model with pmdarima.auto_arima...")
        auto_arima_model = pm.auto_arima(train_close, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore')
        arima_order = auto_arima_model.order
        logger.info(f"Best ARIMA order found: {arima_order}")
        arima_model = ARIMA(train_close, order=arima_order).fit()
        joblib.dump(arima_model, ARIMA_PATH)
        logger.info(f"ARIMA model saved to {ARIMA_PATH}")
        arima_pred_train = arima_model.predict(start=LOOK_BACK, end=split_idx-1)
        if len(arima_pred_train) != len(train_close[LOOK_BACK:split_idx]):
            logger.error(f"ARIMA train prediction length mismatch: {len(arima_pred_train)} vs {len(train_close[LOOK_BACK:split_idx])}")
            return
        residuals_train = train_close[LOOK_BACK:split_idx] - arima_pred_train
        volume_train_aligned = train_volume[LOOK_BACK:split_idx]
        arima_pred_test = arima_model.predict(start=split_idx, end=len(df)-1)
        if len(arima_pred_test) != len(test_close):
            logger.error(f"ARIMA test prediction length mismatch: {len(arima_pred_test)} vs {len(test_close)}")
            return
        residuals_test = test_close - arima_pred_test
        volume_test_aligned = test_volume

        # Stack features: residuals and volume
        train_features = np.column_stack([residuals_train, volume_train_aligned])
        test_features = np.column_stack([residuals_test, volume_test_aligned])

        # Remove NaN rows from features
        train_features, = remove_nan_rows(train_features)
        test_features, = remove_nan_rows(test_features)

        # Scale both features
        scaled_train, scaled_test, scaler = preprocess(train_features, test_features)

        # Create sequences for LSTM
        X_train, y_train = create_sequences(scaled_train)
        X_test, y_test = create_sequences(scaled_test)

        # Remove NaN rows in sequences and targets
        X_train, y_train = remove_nan_rows(X_train, y_train)
        X_test, y_test = remove_nan_rows(X_test, y_test)

        if len(X_train) == 0 or len(X_test) == 0:
            logger.error("Not enough data after sequence creation for LSTM.")
            return

        # --- LSTM ---
        logger.info("Training LSTM on ARIMA residuals and Volume...")
        model = build_lstm((X_train.shape[1], X_train.shape[2]))
        es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[es],
            verbose=2
        )
        # Save LSTM model in HDF5 format only
        model.save(MODEL_PATH)
        logger.info(f"LSTM model saved to {MODEL_PATH}")

        # --- Hybrid Prediction ---
        lstm_pred_scaled = model.predict(X_test)
        # Inverse transform: pad with zeros for the second feature
        lstm_pred = scaler.inverse_transform(np.column_stack([lstm_pred_scaled, np.zeros_like(lstm_pred_scaled)]))[:,0]
        arima_pred_test_seq = arima_pred_test[LOOK_BACK:]
        final_pred = arima_pred_test_seq + lstm_pred
        actual = test_close[LOOK_BACK:]

        # Remove NaNs from predictions and actuals
        mask = ~np.isnan(final_pred) & ~np.isnan(actual)
        final_pred = final_pred[mask]
        actual = actual[mask]

        # Save predictions
        pred_df = pd.DataFrame({
            'Actual': actual,
            'Predicted': final_pred
        })
        pred_df.to_csv(PREDICTIONS_PATH, index=False)
        logger.info(f"Predictions saved to {PREDICTIONS_PATH}")

        # Print metrics
        rmse = np.sqrt(mean_squared_error(actual, final_pred))
        mape = mean_absolute_percentage_error(actual, final_pred)
        logger.info(f"Test RMSE: {rmse:.2f}")
        logger.info(f"Test MAPE: {mape:.2%}")
        logger.info(f"Best Validation Loss: {min(history.history['val_loss']):.6f}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def predict_next(df, n_steps=1):
    # Input validation
    if n_steps < 1:
        logger.error("n_steps must be positive.")
        return None
    try:
        series = get_series(df)
        if len(series) < LOOK_BACK:
            logger.error(f"Not enough data for prediction (need at least {LOOK_BACK} rows)")
            return None
        # Load ARIMA and LSTM
        if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(ARIMA_PATH)):
            logger.error("Trained model, scaler, or ARIMA not found. Please train the model first.")
            return None
        arima_model = joblib.load(ARIMA_PATH)
        lstm_model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        preds = []
        # Use last LOOK_BACK points for ARIMA and LSTM
        for i in range(n_steps):
            # ARIMA forecast for this step
            arima_forecast = arima_model.forecast(steps=1)[0]
            # For the first step, if enough history, use LSTM on residuals
            if i == 0 and len(series) >= LOOK_BACK*2:
                arima_hist = arima_model.predict(start=len(series)-LOOK_BACK, end=len(series)-1)
                residuals_hist = series[-LOOK_BACK:] - arima_hist
                scaled_resid = scaler.transform(residuals_hist.reshape(-1, 1))
                input_seq = scaled_resid.reshape(1, LOOK_BACK, 1)
                lstm_pred_scaled = lstm_model.predict(input_seq)
                lstm_pred = scaler.inverse_transform(np.column_stack([lstm_pred_scaled, np.zeros_like(lstm_pred_scaled)]))[:,0]
                final_pred = arima_forecast + lstm_pred
            else:
                # For future steps, use only ARIMA forecast (no LSTM, as residuals are not available)
                final_pred = arima_forecast
            preds.append(final_pred)
            # Update series for next step
            series = np.append(series, final_pred)
            # ARIMA model is not incrementally updated; we use rolling forecast
        logger.info(f"Next {n_steps} predicted value(s): {preds}")
        return preds
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return None

# --- FASTAPI REST API ---
if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="Crude Oil Price Prediction (Hybrid ARIMA+LSTM)")
    parser.add_argument('--api', action='store_true', help='Run as FastAPI server')
    args, unknown = parser.parse_known_args()

    df = load_data(CSV_FILE_PATH)

    if args.api:
        from fastapi import FastAPI, HTTPException, Header
        from pydantic import BaseModel
        import uvicorn

        app = FastAPI(title="Crude Oil Price Prediction API (Hybrid ARIMA+LSTM)")

        class PredictRequest(BaseModel):
            n_steps: int = 1

        class TrainRequest(BaseModel):
            token: str

        @app.post("/predict")
        def predict_endpoint(req: PredictRequest):
            if req.n_steps < 1:
                raise HTTPException(status_code=400, detail="n_steps must be positive.")
            preds = predict_next(df, n_steps=req.n_steps)
            if preds is None:
                raise HTTPException(status_code=500, detail="Prediction failed or model not trained.")
            return {"predictions": preds}

        @app.post("/train")
        def train_endpoint(req: TrainRequest):
            if req.token != ADMIN_TOKEN:
                raise HTTPException(status_code=403, detail="Unauthorized.")
            try:
                train_and_predict(df)
                return {"status": "Model retrained and predictions updated."}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Training failed: {e}")

        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        train_and_predict(df)
        predict_next(df, n_steps=1)
        predict_next(df, n_steps=5) 