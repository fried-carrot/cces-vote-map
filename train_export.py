# Train on CCES 2006-2024, export logistic coefficients to JS for the interactive map.
# Temporal holdout: train <=2019, test 2020-2024.

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

DTA = "cumulative_2006-2024.dta"

# ---- feature definitions -----------------------------------------------------
ONEHOT = ["educ", "race", "hispanic", "gender", "religion", "marstat", "union",
          "ownhome", "employ", "citizen", "relig_imp", "relig_bornagain", "has_child"]
NUMERIC = ["age", "faminc_ord"]          # standardized
PLACE = ["dist_lean"]                     # target-encoded district Democratic lean
                                          # (EB-shrunk: district -> state -> national)

FAMINC_ORDER = ["Less than 10k", "10k - 20k", "20k - 30k", "30k - 40k", "40k - 50k",
                "50k - 60k", "60k - 70k", "70k - 80k", "80k - 100k", "100k - 120k",
                "120k - 150k", "150k+"]
FAMINC_MAP = {v: i for i, v in enumerate(FAMINC_ORDER)}   # ordinal 0..11

LOAD_COLS = ["year", "voted_pres_party", "state", "cd", "birthyr", "faminc",
             "weight_cumulative"] + ONEHOT

# ---- load --------------------------------------------------------------------
print("Loading .dta ...", flush=True)
df = pd.read_stata(DTA, columns=LOAD_COLS, convert_categoricals=True)
for c in ONEHOT + ["state", "cd", "voted_pres_party", "faminc"]:
    df[c] = df[c].astype("object")
df["cd"] = df["cd"].astype(str)

# target: two-party only
df = df[df["voted_pres_party"].isin(["Democratic", "Republican"])].copy()
df["y"] = (df["voted_pres_party"] == "Democratic").astype(int)

# age from birthyr, clipped to plausible adult range
df["age"] = (df["year"] - df["birthyr"]).clip(18, 100)

# income ordinal; unknown -> NaN (filled with training median later)
df["faminc_ord"] = df["faminc"].map(FAMINC_MAP)

# clean the stray relig_bornagain "8.0" and any NaNs into an explicit Unknown level
for c in ONEHOT:
    df[c] = df[c].where(df[c].notna(), "Unknown")
    df[c] = df[c].apply(lambda v: "Unknown" if isinstance(v, float) else str(v))

df["w"] = df["weight_cumulative"].fillna(1.0).clip(lower=0)
df = df.dropna(subset=["age"])
print(f"Modeling rows: {len(df):,}  Dem share (weighted): "
      f"{np.average(df.y, weights=df.w):.3f}", flush=True)

# ---- temporal split ----------------------------------------------------------
train = df[df["year"] <= 2019].copy()
test = df[df["year"] >= 2020].copy()
print(f"train {len(train):,}  test {len(test):,}", flush=True)

# ---- place feature: hierarchical lean, computed on TRAIN only ----------------
# national -> state (K=500) -> district (K=50). Districts unseen in training
# (post-2022 redistricting created new labels) fall back to their state's lean.
nat = np.average(train.y, weights=train.w)

g = train.groupby("state").apply(
    lambda x: np.average(x.y, weights=x.w), include_groups=False)
n = train.groupby("state")["w"].sum()
K_ST = 500.0
state_lean = ((g * n + nat * K_ST) / (n + K_ST)).to_dict()

cd_state = train.groupby("cd")["state"].first()
gd = train.groupby("cd").apply(
    lambda x: np.average(x.y, weights=x.w), include_groups=False)
nd = train.groupby("cd")["w"].sum()
K_CD = 50.0
prior = cd_state.map(state_lean).fillna(nat)
dist_lean = ((gd * nd + prior * K_CD) / (nd + K_CD)).to_dict()

def place(d):
    st = d["state"].map(state_lean).fillna(nat)
    return d["cd"].map(dist_lean).fillna(st)

for d in (train, test):
    d["dist_lean"] = place(d)

# ---- encode ------------------------------------------------------------------
# fixed one-hot vocab from TRAIN so JS and Python agree exactly
vocab = {c: sorted(train[c].unique().tolist()) for c in ONEHOT}
inc_med = float(train["faminc_ord"].median())
num_mean = {"age": float(train["age"].mean()), "faminc_ord": inc_med}
num_std = {"age": float(train["age"].std()), "faminc_ord": float(train["faminc_ord"].std())}
place_mean = float(train["dist_lean"].mean())
place_std = float(train["dist_lean"].std() + 1e-9)

def build_X(d):
    parts, names = [], []
    for c in ONEHOT:
        for lvl in vocab[c]:
            parts.append((d[c] == lvl).astype(float).values)
            names.append(f"{c}={lvl}")
    for c in NUMERIC:
        v = d[c].fillna(inc_med if c == "faminc_ord" else num_mean[c])
        parts.append(((v - num_mean[c]) / num_std[c]).values)
        names.append(c)
    v = d["dist_lean"]
    parts.append(((v - place_mean) / place_std).values)
    names.append("dist_lean")
    return np.column_stack(parts), names

Xtr, feat_names = build_X(train)
Xte, _ = build_X(test)
ytr, yte = train.y.values, test.y.values
wtr, wte = train.w.values, test.w.values

# ---- primary model: logistic -------------------------------------------------
print("Fitting logistic ...", flush=True)
clf = LogisticRegression(max_iter=2000, C=1.0)
clf.fit(Xtr, ytr, sample_weight=wtr)

def report(name, p_te):
    acc = accuracy_score(yte, (p_te > 0.5).astype(int), sample_weight=wte)
    auc = roc_auc_score(yte, p_te, sample_weight=wte)
    ll = log_loss(yte, p_te, sample_weight=wte)
    print(f"  {name:<28} acc={acc:.4f}  auc={auc:.4f}  logloss={ll:.4f}")
    return acc, auc

print("\n=== TEST metrics (2020-2024) ===")
p_log = clf.predict_proba(Xte)[:, 1]
report("logistic (exported)", p_log)

# ---- benchmark: HistGradientBoosting ----------------------------------------
print("Fitting HistGradientBoosting (benchmark) ...", flush=True)
gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                    max_depth=6, l2_regularization=1.0)
gb.fit(Xtr, ytr, sample_weight=wtr)
report("hist-gradient-boosting", gb.predict_proba(Xte)[:, 1])

# ---- baselines to beat -------------------------------------------------------
report("baseline: dist-lean only", test["dist_lean"].values)
maj = 1 if nat >= 0.5 else 0
acc_maj = accuracy_score(yte, np.full_like(yte, maj), sample_weight=wte)
print(f"  {'baseline: majority-class':<28} acc={acc_maj:.4f}")

# ---- per-district population prediction (map default layer) ------------------
# mean predicted P(Dem) over each district's 2024 respondents = MRP-lite reality map
latest = df[df["year"] == 2024].copy()
latest["dist_lean"] = place(latest)
Xl, _ = build_X(latest)
latest["phat"] = clf.predict_proba(Xl)[:, 1]
dist_pred = latest.groupby("cd").apply(
    lambda x: float(np.average(x.phat, weights=x.w)), include_groups=False).to_dict()
dist_n = latest.groupby("cd").size().to_dict()
state_pred = latest.groupby("state").apply(
    lambda x: float(np.average(x.phat, weights=x.w)), include_groups=False).to_dict()

# ---- export to JS ------------------------------------------------------------
coef = {feat_names[i]: float(clf.coef_[0][i]) for i in range(len(feat_names))}
export = {
    "meta": {
        "target": "P(Democratic two-party presidential vote)",
        "trained_on": "CCES Cumulative 2006-2019", "tested_on": "2020-2024",
        "note": "self-reported party/ideology excluded; district lean is the place-based signal",
    },
    "intercept": float(clf.intercept_[0]),
    "coef": coef,
    "onehot": ONEHOT, "vocab": vocab,
    "numeric": NUMERIC, "num_mean": num_mean, "num_std": num_std,
    "faminc_order": FAMINC_ORDER,
    "dist_lean": {k: float(v) for k, v in dist_lean.items()},
    "state_lean": {k: float(v) for k, v in state_lean.items()},
    "cd_state": {k: str(v) for k, v in cd_state.items()},
    "place_mean": place_mean, "place_std": place_std,
    "dist_pred": dist_pred, "dist_n": dist_n, "state_pred": state_pred,
    "national_dem": float(nat),
}
with open("model_export.js", "w") as f:
    f.write("window.MODEL = " + json.dumps(export) + ";\n")
print("\nWrote model_export.js")
