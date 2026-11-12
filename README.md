# Titanic 특성 공학 비교 과제

빅데이터분석 과제: 891명의 공개 Titanic 데이터로 특성 공학 파이프라인을 비교한 실험입니다.
수치는 모두 실제 실행 결과이며, 합성하거나 예상한 점수가 아닙니다.

## 결과

- 80:20 계층 분할: 학습 712명 / 최종 평가 179명, random_state=42
- 12개 전처리 설정 × Logistic Regression / Random Forest = 24개 실험
- 같은 학습 데이터의 5-fold Stratified CV로 비교, F1을 기준으로 선택
- 선택 결과: 중앙값 대체 + One-Hot + StandardScaler + 파생 변수 + RF 중요도 Top 10 + Random Forest
- GridSearchCV 최종 설정: n_estimators=180, max_depth=8, min_samples_leaf=2
- 최종 평가: Accuracy 0.8268 / Precision 0.8065 / Recall 0.7246 / F1 0.7634 / ROC AUC 0.8522
- 과적합 확인: 변수 선택 전후의 학습-검증 F1 격차 (RF 9.78%p → 8.38%p, LR 0.63%p → 0.64%p)
- SHAP(TreeExplainer)으로 최종 모델의 변수별 기여도 설명

## 파일 구성

- `Titanic_Feature_Engineering_Report.pdf`: 한국어 보고서
- `Titanic_Feature_Engineering.ipynb`: 저장된 실제 결과를 읽어 보여주는 실행된 노트북. RECOMPUTE=True로 전체 실험 재실행 가능
- `analysis.py`: 모든 전처리, EDA, 교차검증, GridSearchCV, 평가, 그래프 생성 코드
- `build_report.py`: 결과 파일로 PDF를 만드는 코드. Windows 맑은 고딕을 사용하며, 없으면 `KOREAN_FONT_DIR` 폴더의 맑은 고딕 또는 나눔고딕(`NanumGothic-Regular.ttf`, `NanumGothic-Bold.ttf`)을 사용
- `create_notebook.py`: 노트북 생성 및 실행 코드
- `data/titanic.csv`: 원본 데이터의 공개 미러 사본
- `results/`: 전체 24개 성능표, fold별 점수, 통제 비교, 원본/선택 변수, 최종 예측, 데이터 해시, 모델
- `figures/`: 보고서에 들어간 그래프 PNG
- `requirements.txt`: 실제 실행 환경의 패키지 버전

## 실행

Python 3.12 이상에서 이 폴더를 작업 경로로 사용합니다.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python analysis.py
python build_report.py --name 이름 --student-id 학번 --department 학과 --github-url https://github.com/OWNER/REPOSITORY
```

`build_report.py`의 인자는 모두 선택 사항이며, 표지와 마지막 쪽에 표시됩니다.

노트북은 VS Code 또는 Jupyter에서 열 수 있습니다. 기본값은 `RECOMPUTE=False`입니다.
기존 결과를 표시하는 셀도 실제 실행되어 저장되었습니다. 실험을 새로 수행하려면 True로 바꾸세요.
데이터가 동봉되어 있으므로 분석 재실행에는 네트워크가 필요하지 않습니다.
`analysis.py`는 원본 CSV의 SHA-256을 검증하고, 변경되면 중단합니다.
패키지 버전·분할·난수 시드가 달라지면 결과가 달라질 수 있습니다.

모델 불러오기 예시(이 폴더에서 실행):

```python
import joblib
import pandas as pd
import analysis

pipeline = joblib.load('results/final_pipeline.joblib')
passengers = pd.read_csv('data/titanic.csv').drop(columns='Survived')
print(pipeline.predict_proba(passengers.head())[:, 1])
```

## 해석상 주의

Base는 결측이 없는 수치형 4개만 사용한 무전처리 기준선입니다. 전체 모델과의 차이는 변수 정보량과 전처리 효과가 함께 반영됩니다.
전처리 하나의 효과는 Ref를 중심으로 하나씩 바꾸는 통제 비교를 사용하세요.
교차검증 점수는 같은 fold로 여러 설정을 선택했으므로 낙관적일 수 있으며, 최종 일반화 성능은 별도의 179명 결과를 보세요.
순열 중요도는 최종 평가 후 설명 용도로만 계산했고 모델 재선택에 사용하지 않았습니다.
이상치는 IQR로 탐지했지만 오류라고 단정하지 않고 유지했습니다. 가족/티켓 단위 그룹 분할은 실시하지 않았습니다.

## 출처와 제출

- 원 데이터/문제: https://www.kaggle.com/competitions/titanic/data
- 사용한 CSV 미러: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
- SHA-256: `4a437fde05fe5264e1701a7387ac6fb75393772ba38bb2c9c566405af5af4bd7`
- Pipeline 참고: https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
- 누수 방지 참고: https://scikit-learn.org/stable/common_pitfalls.html
- 순열 중요도 참고: https://scikit-learn.org/stable/modules/permutation_importance.html

추가 가산점 항목 중 Pipeline, GridSearchCV, SHAP 기반 설명, Feature Importance 시각화를 구현했습니다.
LIME 및 AutoML은 수행하지 않았습니다.
