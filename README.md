#  Intelligent Receipt Processing & Expense Prediction System

An end-to-end Machine Learning and Data Engineering pipeline that ingests raw receipt images, extracts key financial data using OCR and NLP, categorizes expenses, and forecasts future spending using time-series analysis.

## 🚀 Project Overview

This project solves the messy reality of physical document processing. Using the SROIE dataset, the system reads hundreds of scanned receipts, intelligently extracts the Merchant, Date, and Total Amount, and builds a robust data warehouse. Finally, it applies predictive analytics to model financial habits while actively filtering out AI hallucinations and outliers.

##  Key Features

* **Multimodal AI OCR Pipeline:** Ingests receipt images and reads text regardless of language, rotation, or formatting without relying on rigid coordinate bounding boxes.
* **Intelligent Information Extraction:** Uses NLP to locate the true "Total" and "Merchant" names among competing numbers and tax lines.
* **Zero-Shot Classification:** Automatically categorizes extracted receipts into distinct expense buckets (e.g., Groceries, Transport, Dining).
* **Automated Checkpointing:** Processes large image batches safely, ensuring progress isn't lost if the script is interrupted.
* **Predictive Forecasting:** Uses **Facebook Prophet** to map monthly spending trends and forecast future expenses with calculated confidence intervals.

## Data Architecture (Medallion Approach)

To ensure high data quality, this project implements a Medallion Data Architecture:

* **🥉 Bronze Layer (Raw):** 600+ raw, unstructured `.jpg` SROIE receipt images. *(Note: Excluded from this repo to save space).*
* **🥈 Silver Layer (Processed):** `data/extracted_receipts.json` — The raw OCR text mapped to JSON, containing the initial AI extractions.
* **🥇 Gold Layer (Cleaned Analytics):** `data/expenses.db` — A relational SQLite data warehouse. Here, rigorous Data Engineering logic is applied to filter out OCR hallucinations (such as scanning artifacts resulting in $55,000,000 totals) and restrict date boundaries to the verified 2016–2019 window.

## Technology Stack

* **Language:** Python
* **Computer Vision / OCR:** EasyOCR / Multimodal AI
* **Natural Language Processing:** Zero-Shot Classification
* **Database / Data Engineering:** SQLite, SQL, Pandas
* **Machine Learning:** Facebook Prophet
* **Data Visualization:** Matplotlib

## Results & Insights

By applying strict SQL constraints on the Silver Layer data, the time-series model successfully ignores critical OCR errors. The final Prophet forecast model clearly maps out spending density between 2017 and 2018, providing tight confidence intervals without being skewed by billion-dollar outliers or multi-decade date anomalies.

## How to Run (Gold Layer Analytics)

Because the heavy OCR process is already complete, you can query the database or run the forecasting model immediately using the provided files:

1. Clone the repository.
2. Ensure you have the required libraries installed (`pandas`, `sqlite3`, `prophet`, `matplotlib`).
3. Run the forecasting engine:
   ```bash
   python src/forecaster.py
