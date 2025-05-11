# Crude Oil Price Prediction

This project predicts crude oil prices using machine learning and deep learning models. The script fetches the dataset from a public URL, preprocesses the data, trains a model, saves the trained model, and outputs predictions.

## Features
- Loads crude oil price data from a public URL (CSV format)
- Preprocesses and visualizes the data
- Trains a model (LSTM or ARIMA)
- Saves the trained model to disk
- Outputs predictions to the console and to a CSV file

## Requirements
Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

1. **Edit the dataset URL in `main.py` if needed.**
   - The script fetches the dataset directly from a public URL (e.g., a raw GitHub CSV link).
2. **Run the script:**

```bash
python main.py
```

- The script will:
  - Download and load the dataset
  - Preprocess and clean the data
  - Train the model
  - Save the trained model (e.g., `model.h5` or `model.pkl`)
  - Output predictions to `predictions.csv`

## Deployment on Ocean Cloud

1. Upload the following files to your Ocean Cloud project:
   - `main.py`
   - `requirements.txt`
   - (Optional) `README.md`
2. Set the entrypoint to `python main.py`.
3. Make sure the environment has Python 3.8+ and internet access to fetch the dataset.
4. Deploy the project using the Ocean Cloud dashboard or CLI.

## Notes
- The script is self-contained and does not require Docker locally.
- The dataset is fetched from a public URL; no manual upload is needed.
- The trained model and predictions are saved in the working directory.

## Example Dataset URL
You can use a public dataset such as:

```
https://raw.githubusercontent.com/selva86/datasets/master/OilPrices.csv
```

Or update the URL in `main.py` to your preferred source.

---

For any issues, please check your Ocean Cloud logs or contact support.

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

## Deployment on Render.com (No Docker)

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

### 5. Troubleshooting
- Check the **Logs** tab in Render for errors
- Ensure all model files are present in the deployed environment
- If you get a port error, use `$PORT` in your start command

---

**You are now live on Render!** 🎉 