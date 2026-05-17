# Meta Stock Forecasting using TensorFlow

## Project Overview
This repository contains a complete, end-to-end time-series forecasting pipeline built to model and predict the next-day closing prices of Meta Platforms, Inc. (META). 

As an **aspiring Data Analyst / Quant**, my goal with this project wasn't just to "predict the market," but to implement a rigorous, disciplined pipeline for processing sequence data. This project demonstrates how to ingest raw financial data, handle temporal dependencies safely, construct lookback windows, and implement a recurrent architecture using **TensorFlow/Keras**.

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
* **Visualization:** Matplotlib
* **Compute Environment:** GPU Accelerated Runtime (Google Colab T4 Instance)



