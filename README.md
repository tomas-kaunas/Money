# Meta Stock Forecasting using TensorFlow

## Project Overview
This repository contains a complete, end-to-end time-series forecasting pipeline built to model and predict the next-day closing prices of Meta Platforms, Inc. (META). 

As an **aspiring Data Analyst / Quant**, my goal with this project wasn't just to "predict the market," but to implement a rigorous, disciplined pipeline for processing sequence data. This project demonstrates how to ingest raw financial data, handle temporal dependencies safely, construct lookback windows, and implement a recurrent architecture using **TensorFlow/Keras** while avoiding common pitfalls like data leakage.

---

## Key Highlights

* **Live Market Ingestion:** Programmatically pulled historical asset data using the Yahoo Finance API (`yfinance`), formatting structural market metrics (OHLCV) into locally clean, flattening datasets.
* **Feature Normalization Rigor:** Financial data varies wildly over multi-year regimes. I implemented localized feature scaling using a `MinMaxScaler` bound to stabilize gradient steps within the neural network.
* **Lookback Array Engineering:** Transformed raw row data into overlapping three-dimensional tensor structures ($X$ matrices of multi-day lookback sequences matching a $y$ vector of next-day closing values) required for sequential model ingest.
* **Recurrent Sequence Modeling:** Designed and compiled a Long Short-Term Memory (LSTM) Recurrent Neural Network using TensorFlow to capture deep temporal trends and asset momentum properties.
* **Live Inference Validation:** Structured a production-like prediction window to extract the most recent sequence blocks and compute out-of-sample live next-day inference values.

---

## Tools used

* **Language:** Python 3
* **Machine Learning / Deep Learning:** TensorFlow 2.x, Keras
* **Data Manipulation & Transformation:** Pandas, NumPy
* **Market Ingestion:** Yahoo Finance API (`yfinance`)
* **Feature Engineering:** Scikit-Learn (`MinMaxScaler`)
* **Visualization:** Matplotlib, Seaborn
* **Compute Environment:** GPU Accelerated Runtime (Google Colab T4 Instance)

---

## Skills Learned & Quantitative Core Takeaways

* **Avoiding Data Leakage:** Implementing sequence data splitting requires strict temporal boundaries. I learned how to isolate training sets from test sets chronologically so the network never glimpses the "future" during validation.
* **Tuning Sequential Models:** Gained hands-on experience handling the temperamental nature of LSTMs—understanding the trade-offs between sequence lookback lengths, dropout configurations, and learning rates.
* **Metrics That Matter:** Shifted from looking at abstract accuracy scores to tracking optimization via Mean Squared Error (MSE) and Root Mean Squared Error (RMSE), tracking financial error vectors in native dollar units.
* **End-to-End Pipeline Thinking:** Learned that deep learning is 80% data architecture. Building a clean, repeatable workflow from raw data extraction to final visualization is far more valuable than simply stacking complex layers blindly.

