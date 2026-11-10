"""End-to-end churn prediction on the Telco dataset.

python src/pipeline.py   -> reports/metrics.json, reports/*.png, docs/index.html (dashboard)
"""
import base64
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,  # noqa: E402
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


def load_clean(path: Path = ROOT / "data" / "telco_churn.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # TotalCharges has blank strings for brand-new customers (tenure 0) -> numeric, impute 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    df["y"] = (df["Churn"] == "Yes").astype(int)
    return df.drop(columns=["Churn"])


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["n_services"] = sum((d[c] == "Yes").astype(int) for c in SERVICES)
    d["avg_monthly_spend"] = np.where(d["tenure"] > 0, d["TotalCharges"] / d["tenure"].clip(lower=1), d["MonthlyCharges"])
    d["charge_gap"] = d["MonthlyCharges"] - d["avg_monthly_spend"]  # recent price hike signal
    d["tenure_bucket"] = pd.cut(d["tenure"], [-1, 6, 12, 24, 48, 100], labels=["0-6", "7-12", "13-24", "25-48", "49+"]).astype(str)
    d["autopay"] = d["PaymentMethod"].str.contains("automatic").astype(int)
    return d


def build_model(kind: str, X: pd.DataFrame):
    cat = X.select_dtypes(exclude="number").columns.tolist()
    num = X.select_dtypes(include="number").columns.tolist()
    scale = StandardScaler() if kind == "logreg" else "passthrough"
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat), ("num", scale, num)])
    clf = {
        "dummy": DummyClassifier(strategy="prior"),
        "logreg": LogisticRegression(max_iter=2000, C=0.5),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
        "hist_gb": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=250, random_state=SEED),
    }[kind]
    return Pipeline([("pre", pre), ("clf", clf)])


def fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    df = add_features(load_clean())
    X, y = df.drop(columns=["y"]), df["y"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1"}
    results = {}
    for kind in ["dummy", "logreg", "random_forest", "hist_gb"]:
        cvres = cross_validate(build_model(kind, Xtr), Xtr, ytr, cv=cv, scoring=scoring, n_jobs=1)
        results[kind] = {m: [round(float(cvres[f"test_{m}"].mean()), 4), round(float(cvres[f"test_{m}"].std()), 4)]
                         for m in scoring}
        print(kind, {m: v[0] for m, v in results[kind].items()}, flush=True)

    best = max((k for k in results if k != "dummy"), key=lambda k: results[k]["roc_auc"][0])
    model = build_model(best, Xtr).fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]

    # business threshold: missing a churner costs 5x a wasted retention offer.
    # Chosen on out-of-fold predictions of the TRAIN split only; test set just reports the result.
    oof = cross_val_predict(build_model(best, Xtr), Xtr, ytr, cv=cv, method="predict_proba")[:, 1]

    def cost_at(t, probs, truth):
        return 5 * int(((probs < t) & (truth == 1)).sum()) + int(((probs >= t) & (truth == 0)).sum())

    ths = np.linspace(0.05, 0.9, 86)
    cost = [cost_at(t, oof, ytr) for t in ths]
    t_best = float(ths[int(np.argmin(cost))])
    test_cost = cost_at(t_best, p, yte)
    base_cost = 5 * int((yte == 1).sum())  # offer nobody
    all_cost = int((yte == 0).sum())  # offer everyone
    test = {
        "model": best,
        "roc_auc": round(float(roc_auc_score(yte, p)), 4),
        "pr_auc": round(float(average_precision_score(yte, p)), 4),
        "f1_at_0.5": round(float(f1_score(yte, p >= 0.5)), 4),
        "cost_threshold": round(t_best, 2),
        "cost_vs_no_action_pct": round(100 * (1 - test_cost / base_cost), 1),
        "cost_vs_offer_all_pct": round(100 * (1 - test_cost / all_cost), 1),
        "n_test": int(len(yte)), "churn_rate": round(float(y.mean()), 4),
    }

    imp = permutation_importance(model, Xte, yte, scoring="roc_auc", n_repeats=8, random_state=SEED, n_jobs=1)
    order = np.argsort(imp.importances_mean)[::-1][:10]
    top = [(X.columns[i], float(imp.importances_mean[i])) for i in order]

    # charts
    figs = {}
    fig, ax = plt.subplots(figsize=(5, 4))
    for kind in ["logreg", "random_forest", "hist_gb"]:
        m = build_model(kind, Xtr).fit(Xtr, ytr)
        fpr, tpr, _ = roc_curve(yte, m.predict_proba(Xte)[:, 1])
        ax.plot(fpr, tpr, label=f"{kind} ({roc_auc_score(yte, m.predict_proba(Xte)[:, 1]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (held-out test)")
    ax.legend(fontsize=8)
    figs["roc"] = fig_b64(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh([t[0] for t in top][::-1], [t[1] for t in top][::-1], color="#3b6ea5")
    ax.set(title="Permutation importance (drop in ROC-AUC)")
    figs["imp"] = fig_b64(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(ths, cost)
    ax.axvline(t_best, color="r", ls="--", label=f"best threshold {t_best:.2f}")
    ax.set(xlabel="Decision threshold", ylabel="Train out-of-fold cost", title="Cost vs threshold (5:1)")
    ax.legend()
    figs["cost"] = fig_b64(fig)

    seg = df.groupby("Contract")["y"].mean().mul(100).round(1)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(seg.index, seg.values, color="#c0623b")
    ax.set(ylabel="Churn rate %", title="Churn by contract type")
    figs["seg"] = fig_b64(fig)

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"cv": results, "test": test, "top_features": top}, indent=2))

    rows = "".join(f"<tr><td>{k}</td>" + "".join(f"<td>{v[m][0]:.3f} &plusmn; {v[m][1]:.3f}</td>" for m in scoring) + "</tr>"
                   for k, v in results.items())
    html = f"""<!doctype html><meta charset=utf-8><title>Churn analytics</title>
<style>body{{font:15px system-ui;max-width:1000px;margin:24px auto;padding:0 16px}}
.g{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}img{{max-width:100%}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:6px 10px;text-align:left}}
.k{{display:flex;gap:16px;margin:12px 0}}.k div{{background:#f4f6f8;padding:10px 14px;border-radius:6px}}</style>
<h2>Telco churn: cleaning, features, CV models</h2>
<div class=k><div><b>{test['roc_auc']}</b><br>test ROC-AUC ({best})</div>
<div><b>{test['pr_auc']}</b><br>test PR-AUC</div>
<div><b>{test['cost_vs_no_action_pct']}%</b><br>cost cut vs no action at threshold {test['cost_threshold']}</div>
<div><b>{test['churn_rate']:.1%}</b><br>base churn rate</div></div>
<h3>5-fold CV on training split (mean &plusmn; std)</h3>
<table><tr><th>model</th><th>ROC-AUC</th><th>PR-AUC</th><th>F1@0.5</th></tr>{rows}</table>
<div class=g><img src="data:image/png;base64,{figs['roc']}"><img src="data:image/png;base64,{figs['imp']}">
<img src="data:image/png;base64,{figs['cost']}"><img src="data:image/png;base64,{figs['seg']}"></div>"""
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "index.html").write_text(html, encoding="utf-8")
    print(json.dumps(test, indent=2))


if __name__ == "__main__":
    main()
