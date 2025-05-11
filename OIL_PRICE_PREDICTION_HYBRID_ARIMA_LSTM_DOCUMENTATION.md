# Oil Price Prediction: Hybrid ARIMA + LSTM

## 1. Project Motivation & Problem Statement

### What are we trying to solve?
- **Goal:** Predict future crude oil prices as accurately as possible using historical data.
- **Why is this hard?** Oil prices are influenced by complex, nonlinear, and sometimes unpredictable factors (economics, geopolitics, supply/demand shocks).
- **Key Question:** How can we best model both the predictable (linear, stationary) and unpredictable (nonlinear, non-stationary) parts of oil price movements?

## 2. Intuition & Approach

### Intuition
- **Time series data** often has both linear (trend, seasonality) and nonlinear (shocks, regime changes) components.
- **Classical models** (like ARIMA) are great for stationary, linear patterns.
- **Deep learning models** (like LSTM) excel at capturing nonlinear, non-stationary patterns.
- **Hybrid approach:** Use ARIMA for what it does best, and LSTM for what ARIMA misses (the residuals).

### Why Hybrid?
- **No single model** can capture all aspects of complex time series.
- **Hybrid models** combine strengths: ARIMA for structure, LSTM for flexibility.

## 3. Step-by-Step Solution

### Step 1: Data Loading & Exploration
- **Question:** What does the data look like? Are there missing values, outliers, or trends?
- **Intuition:** Always understand your data before modeling.
- **Technical:** Load CSV, parse dates, sort chronologically.

### Step 2: Stationarity Check & Preprocessing
- **Question:** Is the series stationary? (Do mean/variance change over time?)
- **Intuition:** ARIMA requires stationarity; LSTM does not.
- **Technical:**
  - ARIMA will difference the data as needed (auto_arima).
  - Data is split into train/test **before** scaling to prevent data leakage.
  - Scaler is fit only on training data.

### Step 3: ARIMA Modeling (for Stationary/Linear Part)
- **Question:** What linear, autocorrelated structure can we model?
- **Intuition:** ARIMA captures trends, seasonality, and autocorrelation.
- **Technical:**
  - Use `pmdarima.auto_arima` to select the best ARIMA order automatically.
  - Fit ARIMA to the training set.
  - Predict on train/test, compute residuals (actual - ARIMA prediction).

### Step 4: LSTM Modeling (for Nonlinear/Non-Stationary Residuals)
- **Question:** What patterns did ARIMA miss?
- **Intuition:** Residuals contain the "hard" part—nonlinear, non-stationary effects.
- **Technical:**
  - Train LSTM on ARIMA residuals (after scaling and sequence creation).
  - LSTM learns to predict the part ARIMA cannot.

### Step 5: Hybrid Prediction
- **Question:** How do we combine the models for best prediction?
- **Intuition:** ARIMA + LSTM = full signal (linear + nonlinear).
- **Technical:**
  - For the first future step, predict: **ARIMA forecast + LSTM prediction on residuals**.
  - For further steps, use only ARIMA (since true residuals are not available for unseen data).

### Step 6: Evaluation & Metrics
- **Question:** How well does the model perform?
- **Intuition:** Use metrics that reflect real-world error (RMSE, MAPE).
- **Technical:**
  - Compare predictions to actuals on the test set.
  - Log and save results for reproducibility.

### Step 7: API & CLI Usage
- **Question:** How can others use this model?
- **Intuition:** Make it accessible for both programmatic and manual use.
- **Technical:**
  - CLI: `python main.py` trains and predicts.
  - API: `python main.py --api` launches a FastAPI server with `/predict` and `/train` endpoints.

### Step 8: Deployment & Reproducibility
- **Question:** How do we ensure this works in production?
- **Intuition:** Reproducibility and automation are key.
- **Technical:**
  - All dependencies are listed in `requirements.txt`.
  - Model, scaler, and ARIMA objects are saved and loaded as needed.
  - Logging is used for traceability.

## 4. Key Concepts Explained

### Stationarity
- **Definition:** A stationary series has constant mean/variance over time.
- **Why important?** ARIMA assumes stationarity; differencing is used if not.

### ARIMA
- **What it does:** Models linear relationships, trends, and autocorrelation.
- **Strength:** Interpretable, robust for stationary data.
- **Limitation:** Cannot capture nonlinear or regime-shifting patterns.

### LSTM
- **What it does:** Deep learning model for sequences, can learn nonlinear, long-term dependencies.
- **Strength:** Handles non-stationary, nonlinear data.
- **Limitation:** Needs more data, less interpretable.

### Hybrid Modeling
- **Why:** Real-world time series are rarely purely linear or nonlinear.
- **How:** ARIMA for the easy part, LSTM for the hard part (residuals).

## 5. Practical Tips & Questions to Ask Yourself
- **Is my data stationary?** Use ADF test or plot rolling mean/variance.
- **Is there data leakage?** Never fit scalers or models on test data.
- **Are my residuals white noise?** If not, LSTM can help.
- **How do I evaluate?** Use out-of-sample metrics (RMSE, MAPE).
- **How do I deploy?** Use APIs, save models, document everything.

## 6. Example Workflow
1. **Load data** and explore.
2. **Split** into train/test.
3. **Auto-tune ARIMA** and fit to train.
4. **Compute residuals** and train LSTM on them.
5. **Predict**: ARIMA + LSTM for first step, ARIMA only for further steps.
6. **Evaluate** and log results.
7. **Deploy** via CLI or API.

## 7. Why This Approach is Robust
- **Combines strengths** of statistical and deep learning models.
- **Prevents data leakage** and overfitting.
- **Modular and extensible** for future improvements.

## 8. Further Reading & Next Steps
- Try more advanced decompositions (e.g., STL, wavelets).
- Experiment with multi-step hybridization.
- Add more features (exogenous variables) if available.
- Monitor and retrain models as new data arrives.

---

**This documentation is designed to help you not just use the code, but understand the intuition and reasoning behind every step.**

If you have more questions or want to extend the project, keep asking "What is the intuition?" and "What problem am I solving at this step?"—that's how the best time series analysts work! 