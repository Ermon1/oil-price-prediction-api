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