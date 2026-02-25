# Intelligent Receipt Processing & Expense Prediction System

Ever tried to keep track of a shoebox full of crumpled receipts? This project automates exactly that. 

This is an end-to-end Machine Learning and Data Engineering pipeline. It takes messy, real-world receipt images, reads them using AI, figures out what you bought and where, and then uses time-series forecasting to predict your future spending habits. 

### Key Features

* **Reads Messy Receipts:** Uses advanced OCR (Optical Character Recognition) to read text off images, no matter how the receipt is rotated or formatted.
* **Smart Extraction:** It doesn't just read words; it understands them. It uses NLP to find the *actual* "Total" and "Merchant" names, ignoring useless tax lines or phone numbers.
* **Auto-Categorization:** Automatically buckets your expenses (like *Groceries*, *Transport*, or *Dining*) without needing manual tagging.
* **Crash-Proof Processing:** Processing hundreds of images takes time. The system saves its progress as it goes, so a computer crash won't force you to start over.
* **Future Forecasting:** Uses Facebook's Prophet AI to map out your monthly spending trends and predict future expenses.

---

### How the Data is Organized



To make sure the AI's predictions are actually accurate, this project uses a "Medallion" data architecture to slowly clean the data step-by-step:

* **🥉 Bronze Layer (Raw):** 600+ raw, unorganized `.jpg` receipt images from the SROIE dataset *(Note: Excluded from this repo to keep it lightweight!)*.
* **🥈 Silver Layer (Processed):** The `extracted_receipts.json` file. This is the raw text the AI pulled from the images. It's good, but occasionally the AI hallucinates (like reading a barcode as a $55,000,000 charge).
* **🥇 Gold Layer (Cleaned & Ready):** The `expenses.db` database. This is where the magic happens. We use strict SQL rules to filter out those million-dollar AI hallucinations and lock the dates to realistic timeframes. 

---

###  Built With
* **Language:** Python
* **AI & Machine Learning:** EasyOCR, HuggingFace (Zero-Shot NLP), Facebook Prophet
* **Data Engineering:** Pandas, SQLite, SQL
* **Visualization:** Matplotlib

---

###  The Results

By rigorously cleaning the data in the Gold Layer, the forecasting model is protected from crazy AI typos. The final output generates beautiful, interactive graphs that accurately map out spending density from 2017 to 2018 with tight, realistic confidence intervals.

---

### How to Run It

Because the heavy AI image processing is already done and saved in the database, you can jump straight to the analytics!

1. Clone this repository to your computer.
2. Make sure you have the required libraries installed (`pip install pandas sqlite3 prophet matplotlib`).
3. Run the forecasting engine in your terminal:
   ```bash
   python src/forecaster.py
