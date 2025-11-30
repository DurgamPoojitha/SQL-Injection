#!/usr/bin/env python3
"""
show_results_in_chrome.py

Usage examples:
  python show_results_in_chrome.py --csv /mnt/data/results.csv
  python show_results_in_chrome.py --csv /mnt/data/my_results.csv --label-col label --pred-col pred
  python show_results_in_chrome.py --csv /mnt/data/results.csv --prob-col prob_attack --threshold 0.6
  python show_results_in_chrome.py --csv /mnt/data/results.csv --out /tmp/sqli_report.html

What it does:
 - Loads a CSV of results
 - Auto-detects label/pred/prob columns (or uses provided flags)
 - Converts probabilities to binary predictions using threshold (default 0.5)
 - Computes confusion matrix & classification report
 - Saves confusion matrix image and an HTML report
 - Attempts to open the HTML in the system default browser (usually Chrome if default)
"""

import argparse
import os
import sys
import base64
import io
import webbrowser
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# -------------------------
# Helpers
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Create HTML report (confusion matrix + classification report) and open in browser.")
    p.add_argument("--csv", required=True, help="Path to results CSV")
    p.add_argument("--label-col", default=None, help="Column name for true labels (optional)")
    p.add_argument("--pred-col", default=None, help="Column name for predicted labels (optional)")
    p.add_argument("--prob-col", default=None, help="Column name for predicted probability for positive class (optional)")
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold to convert probability to binary pred (default 0.5)")
    p.add_argument("--out", default=None, help="Output HTML path (defaults to ./sqli_report_<timestamp>.html)")
    p.add_argument("--img-out", default=None, help="Output image path (defaults to same folder as HTML)")
    p.add_argument("--open", action="store_true", help="Open the generated HTML in the default browser")
    return p.parse_args()

def candidate_name(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def norm_label_value(v):
    if pd.isna(v):
        return 0
    s = str(v).strip().lower()
    if s in ('1','attack','att','malicious','yes','true','t','positive','pos'):
        return 1
    if s in ('0','benign','safe','no','false','f','negative','neg'):
        return 0
    # fallback numeric conversion
    try:
        f = float(s)
        return 1 if f >= 0.5 else 0
    except:
        return 0

# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    csv_path = args.csv
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')

    # try to find label/pred/prob columns if not given
    cols = list(df.columns)
    label_col = args.label_col
    pred_col = args.pred_col
    prob_col = args.prob_col

    if label_col is None:
        label_col = candidate_name(cols, ['label','Label','true_label','y','target','actual','ground_truth','gt'])
    if pred_col is None:
        pred_col = candidate_name(cols, ['pred','prediction','predicted','y_pred','yhat','y_hat'])
    if prob_col is None:
        prob_col = candidate_name(cols, ['prob','prob_attack','probability','score','score_attack','y_prob','probability_attack'])

    # If we only have a prob column and no pred col, we'll turn prob -> pred using threshold
    # If we have a pred_col that's numeric/float between 0 and 1, treat as prob (unless prob_col provided)
    if pred_col is None and prob_col is None:
        # try to find a numeric column with values in (0,1)
        for c in cols:
            try:
                vals = df[c].dropna().astype(float)
                if vals.between(0,1).all() and vals.mean() > 0 and vals.mean() < 1:
                    prob_col = c
                    break
            except:
                continue

    # Final checks
    if label_col is None:
        # attempt to detect a binary-like column heuristically
        for c in cols:
            vals = set(df[c].dropna().astype(str).str.strip().str.lower().unique())
            if vals.issubset({'0','1','attack','safe','malicious','benign','yes','no','true','false'}):
                label_col = c
                break

    if label_col is None:
        print("Could not detect a true-label column. Provide --label-col.", file=sys.stderr)
        print("Columns in CSV:", cols, file=sys.stderr)
        sys.exit(2)

    if pred_col is None and prob_col is None:
        print("Could not detect a prediction or probability column. Provide --pred-col or --prob-col.", file=sys.stderr)
        print("Columns in CSV:", cols, file=sys.stderr)
        sys.exit(2)

    # Normalize y_true
    y_true = df[label_col].apply(norm_label_value).astype(int)

    # Build y_pred
    if prob_col is not None:
        # use prob -> binary
        try:
            probs = df[prob_col].astype(float)
        except Exception as e:
            print(f"Failed to parse probabilities from column '{prob_col}': {e}", file=sys.stderr)
            sys.exit(2)
        y_pred = (probs >= args.threshold).astype(int)
    else:
        # use pred_col, coerce
        def norm_pred(v):
            if pd.isna(v):
                return 0
            s = str(v).strip().lower()
            if s in ('1','attack','yes','true','t','positive','pos'):
                return 1
            if s in ('0','benign','safe','no','false','f','negative','neg'):
                return 0
            try:
                f = float(s)
                return 1 if f >= 0.5 else 0
            except:
                return 0
        y_pred = df[pred_col].apply(norm_pred).astype(int)

    # Compute metrics
    labels = [0,1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, target_names=['benign(0)','attack(1)'], digits=4)
    rows_count = len(df)

    # Prepare outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_html = args.out if args.out else os.path.abspath(f"sqli_report_{timestamp}.html")
    if args.img_out:
        img_path = args.img_out
    else:
        img_path = os.path.abspath(f"conf_matrix_{timestamp}.png")

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(['benign (0)','attack (1)'])
    ax.set_yticklabels(['benign (0)','attack (1)'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    thresh = cm.max() / 2. if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(img_path, bbox_inches='tight')
    plt.close(fig)

    # embed image base64
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')

    # Write HTML
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Detection Report - {os.path.basename(csv_path)}</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 30px; }}
    h1, h2 {{ color: #222; }}
    pre.report {{ background:#f7f7f7; padding:15px; border-radius:6px; overflow:auto; }}
    .meta {{ margin-top:12px; }}
  </style>
</head>
<body>
  <h1>Detection Report</h1>
  <h2>Source CSV</h2>
  <p><strong>File:</strong> {csv_path}</p>
  <p class="meta"><strong>Label column:</strong> {label_col} &nbsp;|&nbsp; <strong>Pred column:</strong> {pred_col or 'N/A'} &nbsp;|&nbsp; <strong>Prob column:</strong> {prob_col or 'N/A'} &nbsp;|&nbsp; <strong>Threshold:</strong> {args.threshold}</p>

  <h2>Confusion Matrix</h2>
  <img src="data:image/png;base64,{img_b64}" alt="confusion matrix" style="max-width:700px; border:1px solid #ddd;"/>

  <h2>Classification Report</h2>
  <pre class="report">{report}</pre>

  <h2>Summary</h2>
  <ul>
    <li>Rows analyzed: {rows_count}</li>
    <li>True positives (TP): {cm[1,1]}</li>
    <li>True negatives (TN): {cm[0,0]}</li>
    <li>False positives (FP): {cm[0,1]}</li>
    <li>False negatives (FN): {cm[1,0]}</li>
  </ul>

  <p>Generated on {datetime.now().isoformat()}</p>
</body>
</html>
"""
    with open(out_html, "w", encoding="utf-8") as fo:
        fo.write(html)

    print("Report written to:", out_html)
    print("Confusion matrix image:", img_path)
    print("\nClassification report:\n")
    print(report)

    if args.open:
        url = "file://" + os.path.abspath(out_html)
        print(f"\nOpening report in the default browser: {url}")
        try:
            webbrowser.open(url, new=2)
        except Exception as e:
            print("Failed to open browser:", e, file=sys.stderr)

if __name__ == "__main__":
    main()
