# SQL Injection Detection System (Machine Learning + Rule-Based Hybrid)

This project implements a **complete SQL Injection (SQLi) detection system** using:

- ML-based probabilistic classification  
- Rule-based feature extraction (SQL keywords, comments, encoded payloads, hex, Base64, URL decode layers, etc.)
- Logistic Regression classifier + StandardScaler  
- Interactive HTML reports (Confusion Matrix, ROC Curve, PR Curve, Accuracy vs Threshold, CSV preview)

It supports:
- Training a new model  
- Testing a dataset  
- Classifying a single payload  
- Automatic generation of detailed browser-based visualization reports  

---

## 🚀 **Project Overview**

SQL Injection remains one of the most dangerous web vulnerabilities, allowing attackers to modify backend SQL queries through crafted inputs.  
This project builds a **lightweight, explainable, ML-based SQLi detection engine**.

### 🔥 Key Features
1. **Rich Feature Extraction**
   - UNION SELECT
   - SLEEP(), BENCHMARK()
   - OR 1=1 patterns
   - SQL comments (`--`, `/* */`)
   - Hex patterns (`0x...`)
   - URL encoding layers
   - Base64, ROT13, hex decoding detection
   - Length, ASCII ratio, suspicious characters count

2. **Machine Learning Model**
   - Logistic Regression (balanced class-weight)
   - StandardScaler for normalization  
   - 5-fold Stratified Cross Validation  
   - Model stored as:
     - `sqli_model.pkl`
     - `sqli_scaler.pkl`

3. **Dataset Mode**
   - Run thousands of queries
   - Measure accuracy, FP, FN, TP, TN
   - Exports `results.csv`

4. **Interactive Auto-Generated HTML Report**
   After dataset testing, the system auto-generates a report with:
   - Confusion Matrix  
   - ROC Curve + AUC  
   - Precision-Recall Curve + AP  
   - Accuracy vs Threshold graph  
   - Classification Report  
   - CSV data preview with sorting/search (DataTables)

5. **Single Payload Detection**
   ```
   python demofinal.py "1 OR 1=1"
   ```

---

## 📂 **Project Files**

| File | Description |
|------|-------------|
| `traindetector.py` | Extracts features, trains ML model, saves scaler + classifier. |
| `demofinal.py` | Main detection engine + reporting system. |
| `runtest.py` | External testing script for custom datasets. |
| `sql_injection_dataset.csv` | Dataset used for training/testing. |
| `results.csv` | Auto-generated classification output. |

---

## 🧠 **How the System Works**

### 1️⃣ Feature Extraction  
Each payload is processed by `extract_features_single(...)` which produces 20+ engineered features such as:

- `len`
- `ascii_ratio`
- `pat_UNION SELECT`
- `pat_INFORMATION_SCHEMA`
- `has_base64`
- `has_hex`
- `url_decoded_layers`
- `num_sql_chars`
- and others

These features make SQLi attempts more distinguishable than raw text alone.

---

### 2️⃣ Model Training (`traindetector.py`)
✔ Uses Logistic Regression  
✔ Performs 5-fold cross-validation  
✔ Evaluates accuracy and prints classification report  
✔ Saves trained model & scaler:

```
sqli_model.pkl
sqli_scaler.pkl
```

Run training:
```bash
python traindetector.py
```

---

### 3️⃣ Detection (`demofinal.py`)
There are **two modes**:

---

## ▶️ **1. Classify a Single Payload**

Example:
```bash
python demofinal.py "1 OR 1=1"
```

Output:
```json
{
  "action": "BLOCK",
  "prob_attack": 0.9823,
  "pred": 1,
  "features": {...}
}
```

---

## ▶️ **2. Test Entire Dataset & Generate Report**

```bash
python demofinal.py --out results.csv
```

This will:

1. Read `sql_injection_dataset.csv`
2. Predict on all rows
3. Print final metrics:
   - Accuracy
   - False Positives
   - False Negatives
4. Save detailed predictions to:
   ```
   results.csv
   ```
5. Generate an **interactive HTML report**:
   ```
   report_YYYYMMDD_HHMMSS.html
   ```
6. Automatically open the report in your browser

---

## 📊 **Report Features**

The auto-generated report contains:

### 📌 Confusion Matrix  
Visual comparison between predicted vs actual labels.

### 📌 ROC Curve  
Shows model’s ability to distinguish attack vs benign.

### 📌 Precision–Recall Curve  
Highly useful for imbalanced datasets.

### 📌 Accuracy vs Threshold  
Understand performance sensitivity.

### 📌 Classification Report (Precision, Recall, F1-score)

### 📌 CSV Preview (Interactive)
- sortable  
- searchable  
- pagination  
- downloadable full CSV  

---

## 📦 **Installation**

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

---

## 📝 **Troubleshooting**

### ❗ Browser Doesn’t Open
Open the printed file path manually:
```
file:///C:/.../report_202xxxxx.html
```

### ❗ Model/Scaler Not Found
Ensure:
```
sqli_model.pkl
sqli_scaler.pkl
```
are in the same folder as `demofinal.py`.

### ❗ CSV Columns Not Detected
Your script auto-detects:
- `Sentence`, `Query`, `text`, etc.
- `Label`, `class`, `target`, etc.

If needed, specify manually inside `run_on_dataset()`.

---

## 🏁 Final Notes

This project is **production-ready**, fully modular, and can be extended to:

- REST API (FastAPI/Flask)
- Browser extension payload inspection
- WAF (Web Application Firewall) plugin
- Real-time request monitoring
- Auto-retraining using feedback loop

If you'd like, I can generate:
- A full **project report PDF**
- **UML diagrams**
- **Block diagrams**
- **Architecture diagrams**
- A **PowerPoint presentation** summarizing the project
