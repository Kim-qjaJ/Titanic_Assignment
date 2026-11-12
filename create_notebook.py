"""노트북을 생성하고 실제 Python 커널에서 결과 표시 셀을 실행한다."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parent
DEPS=ROOT.parent/".titanic_dependencies"
if DEPS.exists():
    sys.path.insert(0,str(DEPS))
    os.environ["PYTHONPATH"]=os.pathsep.join([str(DEPS),str(ROOT),os.environ.get("PYTHONPATH","")])
os.environ.setdefault("JUPYTER_RUNTIME_DIR",str(ROOT/".cache/jupyter"))
os.environ.setdefault("IPYTHONDIR",str(ROOT/".cache/ipython"))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache/matplotlib"))
os.environ.setdefault("PYTHONUTF8", "1")
import nbformat
from nbclient import NotebookClient


def build():
    cells=[]
    def md(s): cells.append(nbformat.v4.new_markdown_cell(s))
    def code(s): cells.append(nbformat.v4.new_code_cell(s))
    md("""# Titanic 생존 예측: 특성 공학 비교 과제

**빅데이터분석 01분반 · 실제 실행 결과를 포함한 재현용 노트북**

891명 / 12개 설정 × 2개 모델 / 동일 5-fold CV / 179명 최종 평가.
분석의 전체 구현은 함께 제공된 `analysis.py`입니다. 이 노트북은 기본적으로 저장된 실제 실험 결과를 읽습니다.
아래 `RECOMPUTE=True`를 설정하면 모델 학습부터 다시 수행합니다.

목표: 결측값 대체, 인코딩, 스케일링, 파생 변수와 변수 선택이 모델에 미치는 영향을 비교합니다.
생존(1)을 양성으로 정의하고 CV F1으로 모델을 선택합니다. 최종 평가 데이터는 설정 선택에 사용하지 않습니다.
""")
    code("""from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image
import analysis

ROOT = Path(analysis.__file__).resolve().parent
RESULTS = ROOT / 'results'
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 30)
""")
    md("## 1. 데이터 준비와 검증 분리\n원본 CSV는 공개 Titanic 데이터 미러입니다. `load_data()`가 SHA-256을 검증합니다.")
    code("""df, sha = analysis.load_data()
print('데이터 shape:', df.shape)
print('원본 SHA-256:', sha)
display(df.head())
display(pd.read_csv(RESULTS / 'data_dictionary.csv'))
""")
    code("""manifest = json.loads((RESULTS / 'split_manifest.json').read_text(encoding='utf-8'))
train = df[df.PassengerId.isin(manifest['train_ids'])]
print('학습:', len(train), '| 최종 평가:', len(manifest['test_ids']))
print('학습 타깃 분포:')
display(train.Survived.value_counts().sort_index().rename('count'))
""")
    md("""## 2. EDA

그래프는 학습 712명으로만 작성했습니다. 전체 데이터의 결측 수는 위의 데이터 설명 표와 구분합니다.
Cabin은 학습에서 77.53%가 비어 제외했고, Age와 Embarked는 Pipeline 내부에서 처리합니다.
IQR 이상치는 통계적 탐지값이며 잘못된 기록이라고 단정하지 않고 보존했습니다.
""")
    code("""for name in ['05_missingness.png', '01_histograms.png', '02_boxplots.png',
             '04_target_and_groups.png', '03_heatmap.png']:
    display(Image(filename=str(ROOT / 'figures' / name), width=850))
display(pd.read_csv(RESULTS / 'outliers_train.csv'))
""")
    md("""## 3. 특성 공학 파이프라인

- 수치 결측값: Mean / Median / Most Frequent. 범주형 결측값은 최빈값으로 고정.
- 인코딩: OneHotEncoder / OrdinalEncoder(특성별 정수 매핑).
- 수치 스케일: Standard / MinMax / Robust / 미적용.
- 파생 변수: FamilySize=SibSp+Parch+1, FarePerPerson=Fare/FamilySize, IsAlone, 이름에서 추출한 Title.
- 변수 선택: RF 불순도 중요도 상위 10개. 각 학습 fold에서만 적합.
- 모델: Logistic Regression, Random Forest.

Base는 결측이 없는 Pclass/SibSp/Parch/Fare만 사용하는 무전처리 기준선입니다.
Base와 다른 실험의 차이는 정보량 차이도 포함합니다. 한 처리의 효과는 Ref를 기준으로 한 통제 비교에서 판단합니다.
""")
    code("""display(pd.DataFrame(analysis.configs()).fillna(False))
reference = next(c for c in analysis.configs() if c['name'] == 'Ref')
pipeline = analysis.make_pipeline(reference, 'RF')
print(pipeline)
""")
    md("## 4. 실험 실행 또는 저장 결과 읽기\n전체 학습을 재실행하려면 다음 변수만 True로 변경하세요. 재실행은 결과 CSV와 그림을 갱신합니다.")
    code("""RECOMPUTE = False
if RECOMPUTE:
    summary = analysis.run()
else:
    summary = json.loads((RESULTS / 'summary.json').read_text(encoding='utf-8'))
    print('저장된 실제 실행 결과를 사용합니다:', summary['executed_utc'])
print('난수 시드:', summary['seed'])
print('실행 환경:', summary['versions'])
""")
    md("## 5. 필수 조합과 두 모델 비교\n다음 값은 동일 5-fold CV 평균입니다. 표의 SD는 fold 간 표준편차이며 신뢰구간이 아닙니다.")
    code("""cv = pd.read_csv(RESULTS / 'cv_results.csv')
columns = ['experiment', 'model', 'Accuracy', 'Precision', 'Recall', 'F1', 'F1_sd', 'ROC_AUC']
display(cv[cv.experiment.isin(['Base', 'Exp-1', 'Exp-2', 'Exp-3'])][columns].round(4))
display(Image(filename=str(ROOT / 'figures/06_cv_comparison.png'), width=950))
""")
    md("""## 6. 통제 비교

Ref = Median + One-Hot + Standard + 파생 변수 + 변수 선택 없음.
표의 delta_f1은 변경 설정 - Ref이며, 한 요소만 바꿨습니다.
동일 fold의 대응 차이와 그 SD를 저장했습니다. 이 값으로 통계적 유의성을 주장하지 않습니다.
""")
    code("""display(cv[columns].round(4))
controlled = pd.read_csv(RESULTS / 'controlled_comparisons.csv')
display(controlled.round(4))
""")
    md("""## 7. 변수 선택과 설명

20개 변환 특성 중 상위 10개를 선택했습니다. 아래 목록은 학습 데이터 전체 재적합 결과이며,
CV fold에서는 다른 변수가 선택될 수 있습니다.
순열 중요도는 최종 평가 이후 설명을 위해서만 계산했고 재선택에 사용하지 않았습니다.

과적합 여부는 같은 fold의 학습 F1과 검증 F1의 격차(`F1_gap` = 학습 - 검증)로 확인합니다.
""")
    code("""display(pd.read_csv(RESULTS / 'feature_selection.csv').round(4))
display(Image(filename=str(ROOT / 'figures/08_feature_importance.png'), width=1000))
print('변수 선택 전(Ref) / 후(Top-10)의 학습-검증 F1 격차')
display(cv[cv.experiment.isin(['Ref', 'Top-10'])][['experiment', 'model', 'F1_train', 'F1', 'F1_gap', 'F1_gap_sd']].round(4))
""")
    md("""## 8. GridSearchCV와 최종 평가

최고 CV F1 설정을 선택한 뒤 학습 데이터 안에서 GridSearchCV를 수행했습니다.
여러 설정을 같은 fold로 고른 CV 점수는 낙관적일 수 있습니다. 최종 성능은 별도로 남겨 둔 179명에서 평가했습니다.
Bootstrap 구간은 평가 표본을 2,000회 재표집한 분위수이며 모델 선택 및 가족 의존성은 반영하지 않습니다.
""")
    code("""print('선택 설정 / 모델:', summary['selected_experiment'], summary['selected_model'])
print('최적 파라미터:', summary['best_params'])
print('튜닝 후 CV F1:', summary['tuned_cv_f1'])
metrics = pd.DataFrame({'score': summary['holdout'],
                        'CI95_low': {k:v[0] for k,v in summary['holdout_ci95'].items()},
                        'CI95_high': {k:v[1] for k,v in summary['holdout_ci95'].items()}})
display(metrics.round(4))
display(Image(filename=str(ROOT / 'figures/07_holdout_evaluation.png'), width=900))
""")
    md("""## 9. SHAP 기반 설명

최종 Random Forest에 TreeExplainer를 적용해, 선택된 10개 변수가 평가 데이터 179명의 생존(1) 예측 확률에 준 기여를 계산했습니다.
`mean_abs_shap`이 클수록 예측에 미친 영향이 큽니다. `value_shap_corr`은 변수 값과 SHAP 값의 상관으로, 양수면 평가 표본에서 높은 값이 생존 예측에 양의 기여를 하는 경향을 보였다는 뜻입니다(항상 같은 방향이라는 보장은 아닙니다).
평가 완료 후 설명에만 사용했습니다.
""")
    code("""print('기준값(평균 예측 확률):', round(summary['shap']['base_value'], 4))
display(pd.read_csv(RESULTS / 'shap_importance.csv').round(4))
display(Image(filename=str(ROOT / 'figures/09_shap_summary.png'), width=900))
""")
    md("""## 10. 결론

1. **전처리의 우열은 모델에 따라 달랐습니다.** RF에서는 Top 10 선택이 CV F1을 약 3.10%p 높였지만 LR은 약 1.92%p 낮아졌습니다.
2. **One-Hot이 항상 우수한 것은 아닙니다.** LR에서는 정수 매핑보다 좋았고 RF에서는 정수 매핑이 소폭 높았습니다.
3. **변수 선택은 RF에서만 과적합을 조금 줄였습니다.** RF의 학습-검증 F1 격차는 9.78%p → 8.38%p로 줄고 검증 F1도 올랐습니다. LR은 격차가 0.63%p → 0.64%p로 그대로였고 검증 F1은 낮아졌습니다.
4. **스케일링 차이는 작았습니다.** LR의 Standard/Robust/미적용 F1은 같았고, MinMax는 약간 낮았습니다.
5. **파생 변수 4개 묶음의 효과:** LR F1 +3.31%p, RF +0.45%p. 개별 변수 효과는 분리하지 않았습니다. SHAP에서도 파생 변수 FarePerPerson이 3위로 원본 Fare보다 기여가 컸습니다.

최종 Random Forest의 평가 정확도 **82.68%**, F1 **0.7634**, ROC AUC **0.8522**입니다.
작은 단일 평가 표본, 가족/티켓의 그룹 분할 미수행, 여러 설정을 탐색한 선택 편향은 한계입니다.

## 출처와 제출

- [Kaggle Titanic](https://www.kaggle.com/competitions/titanic/data)
- [사용 CSV 미러](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
- [scikit-learn 혼합형 전처리](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html)
- [scikit-learn 데이터 누수 방지](https://scikit-learn.org/stable/common_pitfalls.html)

PDF, 코드, 원본 CSV, 전체 실험표를 함께 제공합니다.
""")
    nb=nbformat.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.12"}})
    path=ROOT/"Titanic_Feature_Engineering.ipynb"
    nbformat.write(nb,path)
    print("Starting notebook kernel", flush=True)
    client=NotebookClient(nb,timeout=300,startup_timeout=30,kernel_name="python3",resources={"metadata":{"path":str(ROOT)}},allow_errors=False)
    client.execute()
    errors=[o for cell in nb.cells if cell.cell_type == "code" for o in cell.outputs if o.output_type == "error"]
    if errors:
        raise RuntimeError("Notebook display errors: " + str([(e.ename,e.evalue) for e in errors]))
    nbformat.write(nb,path)
    print(f"Executed {sum(c.cell_type == 'code' for c in nb.cells)} cells: {path.name}")
    return path


if __name__=="__main__":
    build()
