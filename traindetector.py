import pandas as pd, re, urllib.parse, base64, codecs
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ----------------------------
# Strict ROT13 detection
# ----------------------------
SUSPICIOUS_KEYWORDS = [
    "select","union","insert","update","delete","drop",
    "sleep","benchmark","information_schema","concat",
    "group_concat","outfile","load_file","or","and","--"
]

def try_rot13_strict(s):
    try:
        raw = str(s)
        dec = codecs.decode(raw, "rot_13")
        if dec.lower() == raw.lower():
            return None
        lo = dec.lower()
        for kw in SUSPICIOUS_KEYWORDS:
            if kw in lo:
                return dec
        return None
    except:
        return None

# ----------------------------
# Other helpers
# ----------------------------
def normalize_spaces(s):
    return re.sub(r'\s+', ' ', str(s).strip())

def try_base64_decode(s):
    cand = ''.join(str(s).split())
    if len(cand) < 8 or len(cand) % 4 not in (0,2,3):
        return None
    if re.fullmatch(r"[A-Za-z0-9+/=\s]+", cand) is None:
        return None
    try:
        decoded = base64.b64decode(cand, validate=True).decode("utf-8", errors="ignore")
        if len(decoded) >= 3:
            return decoded
    except:
        return None
    return None

def try_hex_decode(s):
    m = re.search(r"(?i)(?:0x)?([0-9a-f]{8,})", str(s))
    if not m: return None
    hexstr = m.group(1)
    if len(hexstr) % 2 != 0:
        hexstr = "0"+hexstr
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
    (re.compile(r"(?i)(or|and)\s*1\s*=\s*1"), "OR 1=1"),
    (re.compile(r"--|/\*|\*/"), "SQL COMMENT"),
    (re.compile(r";\s*\w"), "SEMICOLON"),
    (re.compile(r"(?i)\binformation_schema\b"), "INFORMATION_SCHEMA"),
    (re.compile(r"(?i)\bsleep\s*\("), "SLEEP"),
    (re.compile(r"(?i)\bbenchmark\s*\("), "BENCHMARK"),
    (re.compile(r"(?i)0x[0-9a-f]{4,}"), "HEX_0X"),
    (re.compile(r"(?i)\bconcat\s*\("), "CONCAT"),
]

def extract_features_single(t):
    s = normalize_spaces(t)
    f = {}
    f['len'] = len(s)
    f['ascii_ratio'] = sum(1 for c in s if 32 <= ord(c) <= 126) / max(1,len(s))
    for pat, name in strong_patterns:
        f['pat_'+name] = 1 if pat.search(s) else 0
    f['comp_eq'] = 1 if re.search(r"(?i)\bor\b.*?=", s) else 0
    f['string_eq'] = 1 if re.search(r"'[^']*' *= *'[^']*'", s) else 0
    f['num_sql_chars'] = sum(s.count(x) for x in ["'",'"',';','--','/*','*/'])
    dec = s
    layers = 0
    for _ in range(3):
        nd = urllib.parse.unquote_plus(dec)
        if nd != dec:
            layers += 1
            dec = nd
    f['url_decoded_layers'] = layers
    f['has_base64'] = 1 if try_base64_decode(s) else 0
    f['has_hex'] = 1 if try_hex_decode(s) else 0
    f['has_rot13'] = 1 if try_rot13_strict(s) else 0
    return f

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("sql_injection_dataset.csv", encoding="utf-16")
texts = df["Sentence"].astype(str).values
labels = df["Label"].astype(str).str.lower().apply(
    lambda x: 1 if x in ("1","attack","malicious","yes","true","t") else 0
).values

# ----------------------------
# Feature matrix
# ----------------------------
X = pd.DataFrame([extract_features_single(t) for t in texts]).fillna(0)

# ----------------------------
# Scale + Train
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = LogisticRegression(
    max_iter=3000,
    solver="saga",
    class_weight="balanced",
    random_state=42
)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_preds = cross_val_predict(clf, X_scaled, labels, cv=kf)

print("CV Accuracy:", accuracy_score(labels, cv_preds))
print(classification_report(labels, cv_preds))

clf.fit(X_scaled, labels)

joblib.dump(clf, "sqli_model.pkl")
joblib.dump(scaler, "sqli_scaler.pkl")
print("Retraining complete! New model saved.")
