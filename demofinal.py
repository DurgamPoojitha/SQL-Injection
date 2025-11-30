#!/usr/bin/env python3
# demofinal.py
# Usage:
#  single payload: python demofinal.py "1 or 1=1 --"
#  dataset run:    python demofinal.py
#  dataset + save: python demofinal.py --out results.csv

import re, urllib.parse, base64, codecs, time, json, joblib, os, sys, argparse
import pandas as pd
import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, accuracy_score
)
import webbrowser
import base64

##############################################
#  REPORTING HELPERS
##############################################

def _norm_label_val(v):
    if pd.isna(v):
        return 0
    s = str(v).strip().lower()
    if s in ('1','attack','att','malicious','yes','true','t','positive','pos'):
        return 1
    return 0

def generate_report(results_csv_path, threshold=0.5, open_browser=True, max_preview_rows=500):
    """
    Generates a polished HTML report for detection results, writing images and an HTML file next
    to the results CSV. Automatically attempts to open the report in the default browser if
    open_browser=True.

    Args:
        results_csv_path (str): path to the results CSV file.
        threshold (float): probability -> label threshold (used when a prob column is found).
        open_browser (bool): whether to open the generated HTML file automatically.
        max_preview_rows (int): max number of CSV rows to embed in the report preview.
    """
    import os, base64, webbrowser
    from datetime import datetime

    # defensive checks
    if not os.path.exists(results_csv_path):
        print("[report] CSV not found:", results_csv_path)
        return

    # load dataframe
    try:
        df = pd.read_csv(results_csv_path, on_bad_lines='skip')
    except Exception:
        df = pd.read_csv(results_csv_path, encoding='latin1', on_bad_lines='skip')

    cols = list(df.columns)

    # --- detect columns ---
    label_col = None
    pred_col = None
    prob_col = None

    for c in ['label','Label','true_label','actual','target','y','class','gt']:
        if c in cols:
            label_col = c
            break

    if label_col is None:
        for c in cols:
            vals = set(df[c].astype(str).str.strip().str.lower().unique())
            if vals.issubset({'0','1','attack','safe','benign','malicious','true','false','yes','no'}):
                label_col = c
                break

    for c in ['pred','prediction','predicted','y_pred','yhat','y_hat']:
        if c in cols:
            pred_col = c
            break

    for c in ['prob','prob_attack','probability','prob_score','score','score_attack','y_prob']:
        if c in cols:
            prob_col = c
            break

    if label_col is None:
        print("[report] Could not detect a label column in the results CSV. Columns found:", cols)
        return

    # --- build y_true, y_pred, probs ---
    def _norm_label_val(v):
        if pd.isna(v):
            return 0
        s = str(v).strip().lower()
        if s in ('1','attack','att','malicious','yes','true','t','positive','pos'):
            return 1
        return 0

    y_true = df[label_col].apply(_norm_label_val).astype(int)

    probs = None
    y_pred = None
    if prob_col and prob_col in df.columns:
        try:
            probs = df[prob_col].astype(float)
            y_pred = (probs >= threshold).astype(int)
        except Exception:
            probs = None

    if probs is None:
        if pred_col and pred_col in df.columns:
            def norm_pred(v):
                try:
                    return 1 if int(float(v)) == 1 else 0
                except:
                    return 1 if str(v).strip().lower() in ('1','attack','yes','true','t') else 0
            y_pred = df[pred_col].apply(norm_pred).astype(int)
        else:
            print("[report] No usable prediction or probability column found in CSV. Columns:", cols)
            return

    # --- compute metrics ---
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    class_rep = classification_report(y_true, y_pred, target_names=['benign(0)','attack(1)'], digits=4)

    # prepare output paths
    base_dir = os.path.dirname(os.path.abspath(results_csv_path)) or '.'
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cm_path  = os.path.join(base_dir, f"confusion_{ts}.png")
    roc_path = os.path.join(base_dir, f"roc_{ts}.png")
    pr_path  = os.path.join(base_dir, f"pr_{ts}.png")
    acc_path = os.path.join(base_dir, f"acc_{ts}.png")
    html_path = os.path.join(base_dir, f"report_{ts}.html")

    # helper to encode images as base64 for embedding
    def _encode_img(path):
        if path and os.path.exists(path):
            return base64.b64encode(open(path, "rb").read()).decode('utf-8')
        return None

    # --- plot confusion matrix ---
    try:
        fig, ax = plt.subplots(figsize=(6,5))
        ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title('Confusion Matrix')
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['benign (0)','attack (1)'])
        ax.set_yticklabels(['benign (0)','attack (1)'])
        thresh = cm.max() / 2. if cm.max() > 0 else 1
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black')
        fig.tight_layout()
        fig.savefig(cm_path, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print("[report] failed to plot confusion matrix:", e)
        cm_path = None

    # --- ROC, PR, Accuracy-vs-threshold (if probs available) ---
    roc_b64 = pr_b64 = acc_b64 = None
    try:
        if probs is not None:
            # ROC
            fpr, tpr, _ = roc_curve(y_true, probs)
            roc_auc_val = auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(6,5))
            ax.plot(fpr, tpr); ax.plot([0,1],[0,1],'--'); ax.set_title(f'ROC Curve (AUC = {roc_auc_val:.4f})')
            ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
            fig.tight_layout(); fig.savefig(roc_path); plt.close(fig)

            # PR
            prec, rec, _ = precision_recall_curve(y_true, probs)
            ap = average_precision_score(y_true, probs)
            fig, ax = plt.subplots(figsize=(6,5))
            ax.plot(rec, prec); ax.set_title(f'Precision-Recall (AP = {ap:.4f})')
            ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
            fig.tight_layout(); fig.savefig(pr_path); plt.close(fig)

            # Accuracy vs threshold
            thresholds = np.linspace(0.0, 1.0, 101)
            accs = [accuracy_score(y_true, (probs >= th).astype(int)) for th in thresholds]
            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(thresholds, accs)
            ax.set_title('Accuracy vs Threshold'); ax.set_xlabel('Threshold'); ax.set_ylabel('Accuracy')
            fig.tight_layout(); fig.savefig(acc_path); plt.close(fig)
    except Exception as e:
        print("[report] failed to plot ROC/PR/accuracy:", e)
        roc_path = pr_path = acc_path = None

    # encode images
    cm_b64  = _encode_img(cm_path)
    roc_b64 = _encode_img(roc_path)
    pr_b64  = _encode_img(pr_path)
    acc_b64 = _encode_img(acc_path)

    # --- prepare CSV preview (limit rows) ---
    try:
        preview_df = df.copy()
        if len(preview_df) > max_preview_rows:
            preview_note = f"Showing first {max_preview_rows} rows (download full CSV below)."
            preview_df = preview_df.head(max_preview_rows)
        else:
            preview_note = f"Showing all {len(preview_df)} rows."
        # sanitize and render table; add DataTables class
        table_html = preview_df.to_html(classes="display stripe hover", index=False, escape=True)
    except Exception as e:
        table_html = "<p class='muted'>Failed to render CSV preview.</p>"
        preview_note = "Preview not available."

    # --- build prettified HTML (DataTables) ---
    csv_fname = os.path.basename(results_csv_path)
    cm_download = os.path.basename(cm_path) if cm_path else ''
    roc_download = os.path.basename(roc_path) if roc_path else ''
    pr_download = os.path.basename(pr_path) if pr_path else ''
    acc_download = os.path.basename(acc_path) if acc_path else ''

    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SQLi Detection Report — {csv_fname}</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css"/>
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
  <style>
    :root{{--bg:#f6f8fa;--card:#fff;--muted:#6b7280;--accent:#0066cc;--mono:Arial,Helvetica,sans-serif;}}
    html,body{{margin:0;padding:0;background:var(--bg);font-family:var(--mono);color:#111}}
    .container{{max-width:1200px;margin:24px auto;padding:18px}}
    header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:18px}}
    header h1{{margin:0;font-size:20px}}
    .meta small{{color:var(--muted)}}
    .grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:14px}}
    .card{{background:var(--card);border-radius:10px;box-shadow:0 6px 18px rgba(15,15,15,0.06);padding:14px}}
    .big-number{{font-size:28px;font-weight:700}}
    .muted{{color:var(--muted)}}
    .charts{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px}}
    pre.report{{white-space:pre-wrap;background:#f3f6fb;padding:12px;border-radius:8px;overflow:auto}}
    .download-links a{{margin-right:8px;text-decoration:none;color:var(--accent);font-weight:600}}
    .table-wrap{{margin-top:14px;background:var(--card);padding:12px;border-radius:10px;box-shadow:0 6px 18px rgba(15,15,15,0.04)}}
    .footer{{margin-top:18px;color:var(--muted);font-size:13px}}
    @media (max-width:900px){{ .grid{{grid-template-columns:repeat(1,1fr)}} .charts{{grid-template-columns:1fr}} }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>SQL Injection Detection Report</h1>
        <div class="meta"><small>Source: {csv_fname} · Generated: {datetime.now().isoformat()}</small></div>
      </div>
      <div class="download-links">
        <a href="file://{os.path.abspath(results_csv_path)}" download>Download CSV</a>
        {f"<a href='file://{os.path.abspath(cm_path)}' download>Confusion PNG</a>" if cm_download else ""}
        {f"<a href='file://{os.path.abspath(roc_path)}' download>ROC PNG</a>" if roc_download else ""}
        {f"<a href='file://{os.path.abspath(pr_path)}' download>PR PNG</a>" if pr_download else ""}
        {f"<a href='file://{os.path.abspath(acc_path)}' download>Acc vs Thres PNG</a>" if acc_download else ""}
      </div>
    </header>

    <div class="grid">
      <div class="card">
        <div class="muted">Total samples</div>
        <div class="big-number">{len(df)}</div>
      </div>
      <div class="card">
        <div class="muted">Accuracy</div>
        <div class="big-number">{((cm[0,0]+cm[1,1]) / max(1, len(df)) * 100):.2f}%</div>
      </div>
      <div class="card">
        <div class="muted">Errors (FP / FN)</div>
        <div class="big-number">{cm[0,1]} / {cm[1,0]}</div>
      </div>
    </div>

    <div class="card">
      <h3 style="margin-top:0">Classification report</h3>
      <pre style="background:#f3f6fb;padding:12px;border-radius:8px">{class_rep}</pre>
    </div>

    <div class="charts">
      <div class="card">
        <h3 style="margin-top:0">Confusion Matrix</h3>
        {f"<img src='data:image/png;base64,{cm_b64}' style='max-width:100%;border-radius:6px'/>" if cm_b64 else "<p class='muted'>Not available</p>"}
      </div>

      <div class="card">
        <h3 style="margin-top:0">ROC & PR</h3>
        {f"<div style='margin-bottom:8px'><strong>ROC Curve</strong><br/><img src='data:image/png;base64,{roc_b64}' style='max-width:100%;border-radius:6px'/></div>" if roc_b64 else "<p class='muted'>ROC not available</p>"}
        {f"<div><strong>Precision–Recall</strong><br/><img src='data:image/png;base64,{pr_b64}' style='max-width:100%;border-radius:6px'/></div>" if pr_b64 else ""}
      </div>
    </div>

    <div class="card" style="margin-top:12px">
      <h3 style="margin-top:0">Accuracy vs Threshold</h3>
      {f"<img src='data:image/png;base64,{acc_b64}' style='max-width:100%;border-radius:6px'/>" if acc_b64 else "<p class='muted'>Not available</p>"}
    </div>

    <div class="table-wrap">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <div><strong>Results preview</strong><div class="muted" style="font-size:12px">{preview_note}</div></div>
        <div><a href="file://{os.path.abspath(results_csv_path)}" download>Download full CSV</a></div>
      </div>

      {table_html}

    </div>

    <div class="footer">
      Tip: use the search box above the table to filter rows. If the report didn't open automatically, open it manually: <code>file://{os.path.abspath(html_path)}</code>
    </div>
  </div>

  <script>
    $(document).ready(function() {{
        $('table.display').DataTable({{
            pageLength: 25,
            lengthMenu: [10, 25, 50, 100],
            responsive: true
        }});
    }});
  </script>
</body>
</html>
"""

    # write HTML
    try:
        with open(html_path, "w", encoding="utf-8") as fo:
            fo.write(html)
        print("[report] HTML written to:", html_path)
    except Exception as e:
        print("[report] failed to write HTML:", e)

    # attempt to open browser
    if open_browser:
        try:
            webbrowser.open("file://" + os.path.abspath(html_path), new=2)
        except Exception as e:
            print("[report] failed to open browser:", e)

# ----------------------------
# Helper functions (same as training)
# ----------------------------
def normalize_spaces(s):
    return re.sub(r"\s+", " ", str(s).strip())

def try_base64_decode(s):
    s = str(s)
    cand = ''.join(s.split())
    if len(cand) < 8 or len(cand) % 4 not in (0, 2, 3):
        return None
    if re.fullmatch(r"[A-Za-z0-9+/=\s]+", s) is None:
        return None
    try:
        decoded = base64.b64decode(cand, validate=True).decode("utf-8", errors="ignore")
        if len(decoded) >= 3:
            return decoded
    except:
        return None
    return None

def try_rot13(s):
    try:
        return codecs.decode(str(s), "rot_13")
    except:
        return None

def try_hex_decode(s):
    s = str(s)
    m = re.search(r"(?i)(?:0x)?([0-9a-f]{8,})", s)
    if not m:
        return None
    hexstr = m.group(1)
    if len(hexstr) % 2 != 0:
        hexstr = "0" + hexstr
    try:
        decoded = bytes.fromhex(hexstr).decode("utf-8", errors="ignore")
        if len(decoded) >= 3:
            return decoded
    except:
        return None
    return None

strong_patterns = [
    (re.compile(r"(?i)\bunion\b.*\bselect\b"), "UNION SELECT"),
    (re.compile(r"(?i)\bselect\b.*\bfrom\b"), "SELECT FROM"),
    (re.compile(r"(?i)\b(insert|update|delete)\b.*\bfrom\b"), "DML FROM"),
    (re.compile(r"(?i)(\bor\b|\band\b)\s+[\'\"]?\s*1\s*=\s*1"), "OR 1=1"),
    (re.compile(r"(?i)\b(or|and)\b\s+true\b"), "OR TRUE"),
    (re.compile(r"--|/\*|\*/"), "SQL COMMENT"),
    (re.compile(r";\s*\w"), "SEMICOLON"),
    (re.compile(r"(?i)\binformation_schema\b"), "INFORMATION_SCHEMA"),
    (re.compile(r"(?i)\bsleep\s*\("), "SLEEP"),
    (re.compile(r"(?i)\bbenchmark\s*\("), "BENCHMARK"),
    (re.compile(r"(?i)0x[0-9a-f]{4,}"), "HEX_0X"),
    (re.compile(r"(?i)\bconcat\s*\("), "CONCAT"),
]

def extract_features_single(sentence):
    s = normalize_spaces(sentence)
    f = {}
    f["len"] = len(s)
    f["ascii_ratio"] = sum(1 for c in s if 32 <= ord(c) <= 126) / max(1, len(s))
    for pat, name in strong_patterns:
        f["pat_" + name] = 1 if pat.search(s) else 0
    f["comp_eq"] = 1 if re.search(r"(?i)\b(or|and)\b\s+[^\s]+\s*=\s*[^\s]+", s) else 0
    f["string_eq"] = 1 if re.search(r"('[^']*'|\"[^\"]*\")\s*=\s*('[^']*'|\"[^\"]*\")", s) else 0
    f["num_sql_chars"] = sum(s.count(x) for x in ["'", "\"", ";", "--", "/*", "*/"])
    dec = s
    layers = 0
    for i in range(3):
        nd = urllib.parse.unquote_plus(dec)
        if nd != dec:
            layers += 1
            dec = nd
    f["url_decoded_layers"] = layers
    f["has_base64"] = 1 if try_base64_decode(s) else 0
    f["has_rot13"] = 1 if try_rot13(s) and try_rot13(s) != s else 0
    f["has_hex"] = 1 if try_hex_decode(s) else 0
    return f

# ----------------------------
# Load model + scaler once
# ----------------------------
MODEL_FILE = "sqli_model.pkl"
SCALER_FILE = "sqli_scaler.pkl"

if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
    print("ERROR: model/scaler files not found. Make sure sqli_model.pkl and sqli_scaler.pkl are in the same folder.")
    sys.exit(1)

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

# ----------------------------
# Predict one payload
# ----------------------------
def predict_one(text, threshold=0.5):
    """
    Returns a dict:
      - action: "BLOCK" or "ALLOW"
      - prob_attack: float [0..1] model probability
      - pred: 1 if attack else 0
      - features: dict of extracted features

    This version aligns feature columns to the scaler's expected feature names
    to avoid "feature names should match" errors.
    """
    feats = extract_features_single(text)
    X = pd.DataFrame([feats]).fillna(0)

    # Align columns to scaler's feature names if available (scikit-learn >= 1.0 stores feature_names_in_)
    try:
        if hasattr(scaler, "feature_names_in_"):
            expected_cols = list(scaler.feature_names_in_)
            # Reindex to expected columns, filling any missing with 0
            X = X.reindex(columns=expected_cols, fill_value=0)
        else:
            # Fallback: if scaler doesn't provide names, try to align by intersection
            # Keep scaler's n_features_in_ if present as a hint
            if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ == X.shape[1]:
                # assume order matches (best-effort)
                pass
            else:
                # Ensure stable column order: sort columns alphabetically (best-effort)
                X = X.reindex(sorted(X.columns), axis=1, fill_value=0)
    except Exception:
        # Last-resort: ensure numeric array
        X = X.reindex(sorted(X.columns), axis=1, fill_value=0)

    # Convert to numeric / numpy array for scaler
    try:
        Xs = scaler.transform(X)
    except Exception:
        # As a fallback, coerce to numeric numpy array (may change semantics slightly)
        try:
            Xs = X.values.astype(float)
        except Exception:
            # If even that fails, produce a zero vector to avoid crash (very conservative)
            Xs = np.zeros((1, getattr(scaler, "n_features_in_", X.shape[1])))

    # Obtain probability
    try:
        prob = float(model.predict_proba(Xs)[0, 1])
    except Exception:
        try:
            dv = float(model.decision_function(Xs)[0])
            prob = 1.0 / (1.0 + float(pow(2.718281828459045, -dv)))
        except Exception:
            prob = 0.0

    pred = 1 if prob >= threshold else 0
    action = "BLOCK" if pred == 1 else "ALLOW"
    return {"action": action, "prob_attack": prob, "pred": pred, "features": feats}


# ----------------------------
# Run entire dataset and print final results (optionally save CSV)
# ----------------------------
def run_on_dataset(csv_file="sql_injection_dataset.csv", out_file=None, threshold=0.5, text_col_name=None, label_col_name=None):
    if not os.path.exists(csv_file):
        print(f"ERROR: dataset file '{csv_file}' not found in folder.")
        sys.exit(1)
    encs = ["utf-16", "utf-8", "latin1"]
    df = None
    for e in encs:
        try:
            df = pd.read_csv(csv_file, encoding=e, on_bad_lines='skip')
            break
        except Exception:
            continue
    if df is None:
        print("ERROR: couldn't read CSV with tried encodings.")
        sys.exit(1)

    # detect text and label columns
    if text_col_name is None:
        for c in df.columns:
            if str(c).lower() in ("sentence", "text", "payload", "input", "query", "data", "raw"):
                text_col_name = c
                break
        if text_col_name is None:
            for c in df.columns:
                if df[c].dtype == object:
                    text_col_name = c
                    break
    if label_col_name is None:
        for c in df.columns:
            if str(c).lower() in ("label", "class", "target", "is_attack", "attack"):
                label_col_name = c
                break
        if label_col_name is None:
            for c in df.columns:
                vals = set(df[c].dropna().astype(str).str.lower().unique())
                if vals.issubset({'0','1','attack','safe','att','malicious','benign','yes','no'}):
                    label_col_name = c
                    break

    if text_col_name is None or label_col_name is None:
        print("ERROR: couldn't auto-detect text/label columns. Columns found:", list(df.columns))
        sys.exit(1)

    texts = df[text_col_name].astype(str).values
    labels_raw = df[label_col_name].values

    def normalize_label(v):
        if pd.isna(v):
            return 0
        s = str(v).strip().lower()
        if s in ('1','attack','att','malicious','yes','true','t'):
            return 1
        return 0

    labels = [normalize_label(v) for v in labels_raw]

    start = time.time()
    preds = []
    probs = []
    feature_list = []
    for t in texts:
        r = predict_one(t, threshold=threshold)
        probs.append(r["prob_attack"])
        preds.append(r["pred"])
        feature_list.append(r["features"])
    end = time.time()

    total = len(texts)
    correct = sum(1 for p,gt in zip(preds, labels) if p == gt)
    tp = sum(1 for p,gt in zip(preds, labels) if p==1 and gt==1)
    tn = sum(1 for p,gt in zip(preds, labels) if p==0 and gt==0)
    fp = sum(1 for p,gt in zip(preds, labels) if p==1 and gt==0)
    fn = sum(1 for p,gt in zip(preds, labels) if p==0 and gt==1)

    accuracy = (correct/total*100) if total>0 else 0.0
    time_taken = end - start

    # print final block
    print("========================================")
    print("FINAL RESULTS")
    print("========================================")
    print(f"Rows Tested:          {total}")
    print(f"Time Taken:           {time_taken:.2f} seconds")
    print(f"Accuracy:             {accuracy:.2f}%")
    print("--------------------")
    print(f"Correctly Identified: {correct}")
    print(f"False Positives:      {fp} (Safe blocked)")
    print(f"False Negatives:      {fn} (Attacks missed)")
    print("========================================")

    # save detailed CSV if requested
    if out_file:
        out_df = pd.DataFrame({
          text_col_name: texts,
          label_col_name: labels,
          "pred": preds,
          "prob_attack": probs,
          "features": [json.dumps(f) for f in feature_list]
        })
        out_df.to_csv("results.csv", index=False)
        print(f"Detailed results saved to: {out_file}")

# ---------------------------------------
# AUTO GENERATE REPORT AFTER SAVING CSV
# ---------------------------------------
        try:
         generate_report(out_file, threshold=0.5, open_browser=True)
        except Exception as e:
         print("[report] Generation failed:", e)

    return {"total": total, "time": time_taken, "accuracy": accuracy, "correct": correct, "fp": fp, "fn": fn, "tp": tp, "tn": tn}

# ----------------------------
# CLI entry point
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SQLi detector runner")
    parser.add_argument("payload", nargs="?", help="Single payload to classify (if omitted dataset mode runs)")
    parser.add_argument("--out", "-o", help="Save detailed dataset results to CSV file (dataset mode only)")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Decision threshold for BLOCK (default 0.5)")
    args = parser.parse_args()

    if args.payload:
        res = predict_one(args.payload, threshold=args.threshold)
        print(json.dumps(res, indent=2))
    else:
        run_on_dataset(out_file=args.out, threshold=args.threshold)
