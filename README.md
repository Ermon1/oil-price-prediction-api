# Oil Price Prediction API

This project provides a FastAPI-based REST API for hybrid ARIMA+LSTM crude oil price prediction.

## Features
- Hybrid ARIMA + LSTM time series forecasting
- Outlier handling and robust preprocessing
- REST API endpoints for prediction and retraining

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies

## Usage
Run locally:
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Deployment on Render.com 

### 1. Prepare Your Files
- `main.py`
- `lstm_model.h5`
- `arima_model.pkl`
- `scaler.save`
- `requirements.txt`

### 2. (Recommended) Use a GitHub Repository
- Push all files above to a new GitHub repo.

### 3. Create a New Web Service on Render
1. Go to [Render.com](https://dashboard.render.com/)
2. Click **"New +"** and select **"Web Service"**
3. Connect your GitHub repo (or upload files manually)
4. Set **Environment** to Python 3
5. Set **Start Command** to:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
6. (Optional) Set up environment variables if needed
7. Click **Deploy**

### 4. Test Your API
After deployment, test with:
```bash
curl -X POST "https://your-app.onrender.com/predict" -H "Content-Type: application/json" -d '{"n_steps": 1}'
```
