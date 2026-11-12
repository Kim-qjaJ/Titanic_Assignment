"""Titanic 특성 공학 과제: 실제 실험, 학습 데이터 EDA, 재현 가능한 결과 저장.

실행: python analysis.py
데이터: data/titanic.csv (공개 Kaggle train.csv 미러, SHA-256 검증)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCAL_DEPS = ROOT.parent / ".titanic_dependencies"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import hashlib
import importlib.metadata
import json
import platform
import time
import urllib.request
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, make_scorer,
)
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler

SEED = 42
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
EXPECTED_SHA256 = "4a437fde05fe5264e1701a7387ac6fb75393772ba38bb2c9c566405af5af4bd7"
FIG = ROOT / "figures"
RES = ROOT / "results"
NUM = ["Age", "Fare", "SibSp", "Parch"]
CAT = ["Pclass", "Sex", "Embarked"]
DERIVED_NUM = ["FamilySize", "FarePerPerson", "IsAlone"]
BASE_COLS = ["Pclass", "SibSp", "Parch", "Fare"]
METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
SCORING = {
    "Accuracy": "accuracy",
    "Precision": make_scorer(precision_score, zero_division=0),
    "Recall": make_scorer(recall_score, zero_division=0),
    "F1": make_scorer(f1_score, zero_division=0),
    "ROC_AUC": "roc_auc",
}


class TitanicFeatures(TransformerMixin, BaseEstimator):
    """행 단위의 고정 공식만 사용. 타깃/전체 데이터 통계는 사용하지 않는다."""
    def __init__(self, engineered=True):
        self.engineered = engineered

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        out = X[NUM + CAT].copy()
        # sklearn 결측치 처리를 위해 문자열 확장 dtype 대신 object + np.nan을 사용한다.
        out["Pclass"] = X["Pclass"].astype(str).astype(object)
        for col in ["Sex", "Embarked"]:
            out[col] = X[col].astype(object).where(X[col].notna(), np.nan)
        if self.engineered:
            out["FamilySize"] = X["SibSp"] + X["Parch"] + 1
            out["FarePerPerson"] = X["Fare"] / out["FamilySize"]
            out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
            title = X["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
            title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
            out["Title"] = title.where(title.isin(["Mr", "Mrs", "Miss", "Master"]), "Rare").astype(object)
        return out


def configs():
    """필수 네 조합과 한 요소씩 바꾸는 통제 비교 8개: 총 12개."""
    ref = dict(imputation="median", encoding="onehot", scaling="standard", selection=False, engineered=True)
    def c(name, **kwargs):
        return dict(name=name, **(ref | kwargs))
    return [
        c("Base", baseline=True, engineered=False, imputation="none", encoding="none", scaling="none"),
        c("Exp-1", imputation="mean"),
        c("Exp-2", encoding="ordinal", scaling="minmax", selection=True),
        c("Exp-3", imputation="most_frequent", scaling="robust", selection=True),
        c("Ref"),
        c("No-FE", engineered=False),
        c("Ordinal", encoding="ordinal"),
        c("MinMax", scaling="minmax"),
        c("Robust", scaling="robust"),
        c("No-Scale", scaling="none"),
        c("Top-10", selection=True),
        c("Mode", imputation="most_frequent"),
    ]


def make_pipeline(config, model_name):
    if model_name == "LR":
        model = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=SEED)
    else:
        model = RandomForestClassifier(n_estimators=180, max_depth=6, min_samples_leaf=2,
                                       random_state=SEED, n_jobs=1)
    if config.get("baseline"):
        return Pipeline([
            ("raw_numeric", ColumnTransformer([("num", "passthrough", BASE_COLS)], remainder="drop")),
            ("model", model),
        ])
    scaler = {"standard": StandardScaler(), "minmax": MinMaxScaler(),
              "robust": RobustScaler(), "none": "passthrough"}[config["scaling"]]
    nums = NUM + (DERIVED_NUM if config["engineered"] else [])
    cats = CAT + (["Title"] if config["engineered"] else [])
    encoder = (OneHotEncoder(handle_unknown="ignore", sparse_output=False)
               if config["encoding"] == "onehot" else
               OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy=config["imputation"])), ("scale", scaler)]), nums),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", encoder)]), cats),
    ], remainder="drop", verbose_feature_names_out=False)
    selector = (SelectFromModel(
        RandomForestClassifier(n_estimators=120, max_depth=6, min_samples_leaf=2, random_state=SEED, n_jobs=1),
        threshold=-np.inf, max_features=10,
    ) if config["selection"] else "passthrough")
    return Pipeline([
        ("features", TitanicFeatures(config["engineered"])), ("preprocess", prep),
        ("select", selector), ("model", model),
    ])


def load_data():
    path = ROOT / "data" / "titanic.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != EXPECTED_SHA256:
        raise ValueError("데이터 SHA-256이 다릅니다. 원본 및 결과 재현성을 확인하세요: " + sha)
    df = pd.read_csv(path)
    assert df.shape == (891, 12), df.shape
    assert df.PassengerId.is_unique and set(df.Survived.unique()) == {0, 1}
    assert not df[BASE_COLS].isna().any().any()
    return df, sha


def save_json(value, path):
    def convert(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert), encoding="utf-8")


def style_plots():
    sns.set_theme(style="whitegrid", palette=["#147D92", "#ED9E54", "#526987"])
    # 한글 글꼴: KOREAN_FONT_DIR(맑은 고딕 또는 나눔고딕) → Windows 기본 경로 순으로 찾는다.
    folders = [Path(os.environ["KOREAN_FONT_DIR"])] if os.environ.get("KOREAN_FONT_DIR") else []
    folders.append(Path("C:/Windows/Fonts"))
    for fpath in (folder / name for folder in folders for name in ("malgun.ttf", "NanumGothic-Regular.ttf")):
        if fpath.exists():
            font_manager.fontManager.addfont(str(fpath))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(fpath)).get_name()
            break
    plt.rcParams.update({"axes.unicode_minus": False, "figure.dpi": 130,
                         "savefig.dpi": 180, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})


def save_fig(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def eda(df, train):
    # EDA와 시각화는 학습 712명만 사용한다. 전체 데이터의 기본 구조와 결측 수는 별도 표시.
    missing = pd.DataFrame({
        "column": df.columns, "dtype": [str(t) for t in df.dtypes],
        "missing_all": df.isna().sum().values,
        "missing_all_pct": (df.isna().mean() * 100).values,
        "missing_train": train.isna().sum().values,
        "missing_train_pct": (train.isna().mean() * 100).values,
        "unique_all": df.nunique().values,
    })
    missing.to_csv(RES / "data_dictionary.csv", index=False, encoding="utf-8-sig")
    rows = []
    for col in ["Age", "Fare", "SibSp", "Parch"]:
        q1, q3 = train[col].quantile([.25, .75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        mask = (train[col] < lo) | (train[col] > hi)
        rows.append(dict(feature=col, q1=q1, q3=q3, lower=lo, upper=hi,
                         outliers=int(mask.sum()), outlier_pct=float(mask.mean()*100),
                         min=float(train[col].min()), max=float(train[col].max())))
    pd.DataFrame(rows).to_csv(RES / "outliers_train.csv", index=False, encoding="utf-8-sig")
    train.describe().to_csv(RES / "descriptive_statistics_train.csv", encoding="utf-8-sig")
    train.groupby("Sex").Survived.agg(["count", "mean"]).to_csv(RES / "survival_by_sex_train.csv", encoding="utf-8-sig")
    train.groupby("Pclass").Survived.agg(["count", "mean"]).to_csv(RES / "survival_by_class_train.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    sns.histplot(data=train, x="Age", hue="Survived", bins=24, multiple="layer", ax=ax[0], alpha=.55)
    ax[0].set_title("나이 분포 | 결측값 제외")
    sns.histplot(data=train, x="Fare", bins=35, ax=ax[1], color="#147D92")
    ax[1].set_title("운임 분포 | 오른쪽 꼬리가 긴 분포")
    save_fig(fig, "01_histograms.png")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2), layout="constrained")
    sns.boxplot(data=train, x="Survived", y="Age", hue="Survived", legend=False, ax=ax[0])
    sns.boxplot(data=train, x="Survived", y="Fare", hue="Survived", legend=False, ax=ax[1])
    ax[0].set_title("생존 여부에 따른 나이")
    ax[1].set_title("생존 여부에 따른 운임 | 원 단위")
    save_fig(fig, "02_boxplots.png")

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    corr_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
    corr = train[corr_cols].corr()
    corr.to_csv(RES / "correlation_train.csv", encoding="utf-8-sig")
    sns.heatmap(corr, annot=True, fmt=".2f", vmin=-1, vmax=1, center=0, cmap="BrBG", ax=ax, square=True)
    ax.set_title("수치형 변수의 Pearson 상관계수 | 학습 데이터")
    save_fig(fig, "03_heatmap.png")

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3), layout="constrained")
    sns.countplot(data=train, x="Survived", hue="Survived", legend=False, ax=ax[0])
    ax[0].set_title("타깃 분포")
    for bar in ax[0].patches:
        ax[0].annotate(f"{int(bar.get_height())}", (bar.get_x()+bar.get_width()/2, bar.get_height()),
                       ha="center", va="bottom", xytext=(0, 3), textcoords="offset points")
    sns.barplot(data=train, x="Sex", y="Survived", errorbar=None, hue="Sex", legend=False, ax=ax[1])
    sns.barplot(data=train, x="Pclass", y="Survived", errorbar=None, hue="Pclass", legend=False, ax=ax[2])
    for a, title in zip(ax[1:], ["성별 생존 비율", "객실 등급별 생존 비율"]):
        a.set_ylim(0, 1)
        a.set_title(title)
    save_fig(fig, "04_target_and_groups.png")

    fig, ax = plt.subplots(figsize=(8, 2.8), layout="constrained")
    m = missing.query("missing_train > 0").sort_values("missing_train_pct")
    ax.barh(m.column, m.missing_train_pct, color="#147D92")
    ax.set_xlim(0, 100)
    ax.set_xlabel("학습 데이터 결측 비율 (%)")
    for i, row in enumerate(m.itertuples()):
        ax.text(row.missing_train_pct+1, i, f"{row.missing_train_pct:.1f}% ({row.missing_train}명)", va="center")
    save_fig(fig, "05_missingness.png")
    return dict(outliers=rows, duplicated_rows=int(df.duplicated().sum()),
                target_all=df.Survived.value_counts().sort_index().to_dict(),
                target_train=train.Survived.value_counts().sort_index().to_dict(),
                survival_by_sex=train.groupby("Sex").Survived.mean().to_dict(),
                survival_by_class=train.groupby("Pclass").Survived.mean().to_dict())


def metric_values(y, pred, prob):
    return dict(Accuracy=accuracy_score(y, pred), Precision=precision_score(y, pred, zero_division=0),
                Recall=recall_score(y, pred, zero_division=0), F1=f1_score(y, pred, zero_division=0),
                ROC_AUC=roc_auc_score(y, prob))


def run():
    started = time.time()
    for p in [FIG, RES]:
        p.mkdir(parents=True, exist_ok=True)
    style_plots()
    df, sha = load_data()
    X, y = df.drop(columns="Survived"), df.Survived
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)
    assert not set(Xtr.index) & set(Xte.index)
    assert len(Xtr)+len(Xte) == len(df)
    cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(Xtr, ytr))
    save_json({"seed": SEED, "train_ids": Xtr.PassengerId.tolist(), "test_ids": Xte.PassengerId.tolist(),
               "cv": [{"fit_ids": Xtr.iloc[a].PassengerId.tolist(), "validation_ids": Xtr.iloc[b].PassengerId.tolist()} for a,b in cv]},
              RES / "split_manifest.json")
    eda_result = eda(df, df.loc[Xtr.index])
    print(f"Data={df.shape}; train={len(Xtr)}; holdout={len(Xte)}; seed={SEED}", flush=True)
    all_configs = configs()
    save_json(all_configs, RES / "experiment_configs.json")
    summary, folds = [], []
    for config in all_configs:
        for model in ["LR", "RF"]:
            pipe = make_pipeline(config, model)
            scores = cross_validate(pipe, Xtr, ytr, scoring=SCORING, cv=cv, n_jobs=1, error_score="raise",
                                    return_train_score=True)
            row = dict(experiment=config["name"], model=model)
            for metric in METRICS:
                row[metric] = float(np.mean(scores["test_"+metric]))
                row[metric+"_sd"] = float(np.std(scores["test_"+metric], ddof=1))
            # 과적합 확인용: 같은 fold의 학습 점수와 검증 점수 차이(학습 - 검증)
            row["F1_train"] = float(np.mean(scores["train_F1"]))
            row["F1_gap"] = float(np.mean(scores["train_F1"] - scores["test_F1"]))
            row["F1_gap_sd"] = float(np.std(scores["train_F1"] - scores["test_F1"], ddof=1))
            row["fit_seconds"] = float(np.mean(scores["fit_time"]))
            summary.append(row)
            for i in range(5):
                folds.append(dict(experiment=config["name"], model=model, fold=i+1,
                                  **{m: float(scores["test_"+m][i]) for m in METRICS},
                                  F1_train=float(scores["train_F1"][i])))
            print(f"{config['name']:9s} {model} CV F1={row['F1']:.4f} AUC={row['ROC_AUC']:.4f}", flush=True)
    table = pd.DataFrame(summary)
    table.to_csv(RES / "cv_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(folds).to_csv(RES / "cv_fold_results.csv", index=False, encoding="utf-8-sig")

    # 동일 fold 차이: 여러 요인이 동시에 달라지는 필수 조합만으로 효과를 단정하지 않는다.
    pairs = {"Mean vs Median": "Exp-1", "Mode vs Median": "Mode", "Ordinal vs OneHot": "Ordinal",
             "MinMax vs Standard": "MinMax", "Robust vs Standard": "Robust", "No scale vs Standard": "No-Scale",
             "Top10 vs All": "Top-10", "No FE vs FE": "No-FE"}
    fold_df = pd.DataFrame(folds)
    ablation = []
    for model in ["LR", "RF"]:
        ref = fold_df.query("experiment == 'Ref' and model == @model").set_index("fold")
        for label, exp in pairs.items():
            val = fold_df.query("experiment == @exp and model == @model").set_index("fold")
            d = val.F1-ref.F1
            ablation.append(dict(comparison=label, model=model, experiment=exp,
                                 delta_f1=float(d.mean()), delta_sd=float(d.std(ddof=1))))
    pd.DataFrame(ablation).to_csv(RES / "controlled_comparisons.csv", index=False, encoding="utf-8-sig")

    chosen = table.sort_values(["F1", "ROC_AUC"], ascending=False).iloc[0]
    cfg = next(c for c in all_configs if c["name"] == chosen.experiment)
    model_name = chosen.model
    grid = ({"model__C": [.1, .3, 1., 3., 10.]} if model_name == "LR" else
            {"model__max_depth": [4, 6, 8, None], "model__min_samples_leaf": [1, 2, 4]})
    print(f"GridSearchCV on CV-selected {chosen.experiment}/{model_name}", flush=True)
    search = GridSearchCV(make_pipeline(cfg, model_name), grid, scoring=SCORING, refit="F1",
                          cv=cv, n_jobs=1, error_score="raise", return_train_score=False)
    search.fit(Xtr, ytr)
    pd.DataFrame(search.cv_results_).to_csv(RES / "grid_search.csv", index=False, encoding="utf-8-sig")
    final = search.best_estimator_
    # 최종 평가: 설정과 하이퍼파라미터를 확정한 후 홀드아웃 성능을 측정한다.
    prob = final.predict_proba(Xte)[:,1]
    pred = final.predict(Xte)
    metrics = metric_values(yte, pred, prob)
    cm = confusion_matrix(yte, pred, labels=[0,1])
    pd.DataFrame({"PassengerId": Xte.PassengerId, "actual": yte,
                  "prediction": pred, "probability_survived": prob}).to_csv(RES / "holdout_predictions.csv", index=False)
    # 행 단위 bootstrap의 불확실성. 가족/동일 티켓 의존성은 반영하지 않는다.
    rng = np.random.default_rng(SEED)
    boots = {m: [] for m in METRICS}
    ya = yte.to_numpy()
    for _ in range(2000):
        idx = rng.integers(0, len(ya), len(ya))
        if len(np.unique(ya[idx])) < 2:
            continue
        for key, value in metric_values(ya[idx], pred[idx], prob[idx]).items():
            boots[key].append(value)
    ci = {m: np.quantile(v, [.025,.975]).tolist() for m,v in boots.items()}

    # Top-10 실험의 선택 변수를 학습 전체 재적합 모델로 설명한다.
    selection_pipe = make_pipeline(next(c for c in all_configs if c["name"]=="Top-10"), "RF").fit(Xtr,ytr)
    names = selection_pipe.named_steps["preprocess"].get_feature_names_out()
    selector = selection_pipe.named_steps["select"]
    ranks = pd.DataFrame({"feature":names, "importance":selector.estimator_.feature_importances_,
                          "selected":selector.get_support()}).sort_values("importance",ascending=False)
    ranks.to_csv(RES / "feature_selection.csv", index=False, encoding="utf-8-sig")
    # 최종 모델을 수정하지 않고 평가 후 설명에만 사용한다.
    perm = permutation_importance(final, Xte, yte, scoring="roc_auc", n_repeats=30, random_state=SEED, n_jobs=1)
    importance = pd.DataFrame({"feature":Xte.columns, "importance":perm.importances_mean,
                               "sd":perm.importances_std}).sort_values("importance",ascending=False)
    importance.to_csv(RES / "permutation_importance.csv", index=False, encoding="utf-8-sig")
    joblib.dump(final, RES / "final_pipeline.joblib")

    # SHAP: 최종 Random Forest가 실제로 입력받는 변수(전처리·선택 이후) 기준의 기여도.
    # 모델을 바꾸지 않고 평가 데이터 179명의 예측을 설명하는 데에만 사용한다.
    import shap
    final_names = final.named_steps["preprocess"].get_feature_names_out()[final.named_steps["select"].get_support()]
    Xte_model = pd.DataFrame(final[:-1].transform(Xte), columns=final_names, index=Xte.index)
    explainer = shap.TreeExplainer(final.named_steps["model"])
    shap_raw = np.asarray(explainer.shap_values(Xte_model))
    # 이진 분류: 생존(1) 클래스의 기여도만 사용한다. shap 버전에 따라 축 순서가 다르다.
    shap_pos = shap_raw[1] if shap_raw.shape[0] == 2 and shap_raw.ndim == 3 and shap_raw.shape[1] == len(Xte_model) else shap_raw[..., 1]
    base_value = float(np.ravel(explainer.expected_value)[1])
    additivity_error = float(np.abs(base_value + shap_pos.sum(axis=1) - prob).max())
    assert additivity_error < 1e-6, additivity_error
    shap_table = pd.DataFrame({
        "feature": final_names,
        "mean_abs_shap": np.abs(shap_pos).mean(axis=0),
        "mean_shap": shap_pos.mean(axis=0),
        # 변수 값과 기여도의 상관: 양수면 평가 표본에서 높은 값이 양의 기여를 하는 경향(단조 관계 보장은 아님)
        "value_shap_corr": [float(np.corrcoef(Xte_model[c], shap_pos[:, i])[0, 1]) for i, c in enumerate(final_names)],
    }).sort_values("mean_abs_shap", ascending=False)
    shap_table.to_csv(RES / "shap_importance.csv", index=False, encoding="utf-8-sig")
    shap.summary_plot(shap_pos, Xte_model, show=False, plot_size=(9, 4.6), max_display=10)
    fig = plt.gcf()
    fig.axes[0].set_xlabel("SHAP 값 | 생존(1) 예측 확률에 대한 기여")
    fig.axes[0].set_title("최종 모델 SHAP 요약 | 평가 데이터 179명", fontsize=11)
    save_fig(fig, "09_shap_summary.png")

    fig, ax = plt.subplots(figsize=(10,4.4),layout="constrained")
    order = [c["name"] for c in all_configs]
    for off, model, color in [(-.17,"LR","#147D92"),(.17,"RF","#ED9E54")]:
        part = table.query("model == @model").set_index("experiment").loc[order]
        ax.bar(np.arange(len(order))+off,part.F1,width=.32,label=model,color=color,
               yerr=part.F1_sd,capsize=2,error_kw={"linewidth":.8})
    ax.set_xticks(np.arange(len(order)),order,rotation=35,ha="right")
    ax.set_ylim(0,1)
    ax.set_ylabel("5-fold CV F1 | 오차 막대: fold 표준편차")
    ax.legend(loc="lower right")
    save_fig(fig,"06_cv_comparison.png")

    fig, ax = plt.subplots(1,2,figsize=(10,4),layout="constrained")
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",cbar=False,ax=ax[0],square=True,
                xticklabels=["사망(0)","생존(1)"],yticklabels=["사망(0)","생존(1)"])
    ax[0].set(xlabel="예측",ylabel="실제",title="최종 모델 혼동행렬 | 179명")
    fpr,tpr,_ = roc_curve(yte,prob)
    ax[1].plot(fpr,tpr,color="#147D92",label=f"AUC = {metrics['ROC_AUC']:.3f}",lw=2)
    ax[1].plot([0,1],[0,1],"--",color="#9AA7B4")
    ax[1].set(xlabel="False positive rate",ylabel="True positive rate",title="최종 모델 ROC",xlim=(0,1),ylim=(0,1.02))
    ax[1].legend(loc="lower right")
    save_fig(fig,"07_holdout_evaluation.png")

    fig, ax = plt.subplots(1,2,figsize=(11,5),layout="constrained")
    r = ranks.head(15).iloc[::-1]
    ax[0].barh(r.feature,r.importance,color=["#147D92" if v else "#C9D3DA" for v in r.selected])
    ax[0].set(title="변수 선택: RF 불순도 중요도",xlabel="학습 데이터 | 파란 막대 = 선택된 Top 10")
    p = importance.head(8).iloc[::-1]
    ax[1].barh(p.feature,p.importance,xerr=p.sd,color="#ED9E54",capsize=2)
    ax[1].axvline(0,color="#566575",lw=.7)
    ax[1].set(title="최종 모델: 순열 중요도",xlabel="평가 데이터 ROC AUC 감소량 | 30회 평균 ± SD")
    save_fig(fig,"08_feature_importance.png")

    versions = {p:importlib.metadata.version(p) for p in ["numpy","pandas","scipy","scikit-learn","shap","matplotlib","seaborn","joblib","reportlab","pypdf","nbformat","nbclient","ipykernel"]}
    result = dict(dataset_shape=list(df.shape),data_url=DATA_URL,sha256=sha,seed=SEED,
                  train_n=len(Xtr),test_n=len(Xte),cv_folds=5,eda=eda_result,
                  selected_experiment=str(chosen.experiment),selected_model=str(model_name),
                  selected_config=cfg,untuned_cv_f1=float(chosen.F1),
                  best_params=search.best_params_,tuned_cv_f1=float(search.best_score_),
                  holdout=metrics,holdout_ci95=ci,confusion_matrix=cm.tolist(),
                  selected_features=ranks.loc[ranks.selected,"feature"].tolist(),
                  transformed_features_before=len(names),transformed_features_after=int(ranks.selected.sum()),
                  shap=dict(base_value=base_value, additivity_max_error=additivity_error,
                            features=shap_table.to_dict(orient="records")),
                  versions=versions,python=platform.python_version(),
                  executed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=round(time.time()-started,2))
    save_json(result,RES / "summary.json")
    (ROOT/"requirements.txt").write_text("\n".join(f"{k}=={v}" for k,v in versions.items())+"\n",encoding="utf-8")
    print("FINAL " + json.dumps(metrics),flush=True)
    print(f"Completed in {result['elapsed_seconds']} sec",flush=True)
    return result


if __name__ == "__main__":
    # joblib 파일이 다른 Python 프로세스에서도 import 가능한 모듈명을 저장하도록 한다.
    from analysis import run as run_analysis
    run_analysis()
