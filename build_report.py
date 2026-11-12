"""저장된 실험 결과를 읽어 한국어 PDF 보고서를 생성한다. 모델은 재학습하지 않는다."""
from pathlib import Path
import os
import sys
ROOT = Path(__file__).resolve().parent
if (ROOT.parent / ".titanic_dependencies").exists():
    sys.path.insert(0, str(ROOT.parent / ".titanic_dependencies"))
import argparse
import json
from datetime import datetime, timedelta
from xml.sax.saxutils import escape
import pandas as pd
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from PIL import Image as PILImage

FONT_DIR = Path(os.environ.get("KOREAN_FONT_DIR", "C:/Windows/Fonts"))
# 맑은 고딕이 없으면 같은 폴더의 나눔고딕을 사용한다.
for regular, bold in [("malgun.ttf", "malgunbd.ttf"), ("NanumGothic-Regular.ttf", "NanumGothic-Bold.ttf")]:
    if (FONT_DIR / regular).exists() and (FONT_DIR / bold).exists():
        pdfmetrics.registerFont(TTFont("Korean", str(FONT_DIR / regular)))
        pdfmetrics.registerFont(TTFont("KoreanBold", str(FONT_DIR / bold)))
        break
else:
    raise FileNotFoundError(f"한글 글꼴을 찾지 못했습니다: {FONT_DIR}. KOREAN_FONT_DIR을 지정하세요.")
pdfmetrics.registerFontFamily("Korean", normal="Korean", bold="KoreanBold", italic="Korean", boldItalic="KoreanBold")
NAVY = colors.HexColor("#173047")
TEAL = colors.HexColor("#147D92")
MUTED = colors.HexColor("#5A6B7A")
PALE = colors.HexColor("#EDF5F7")
INK = colors.HexColor("#243748")
WIDTH = A4[0] - 88
ST = {
    "body": ParagraphStyle("body", fontName="Korean", fontSize=9.1, leading=14.2, textColor=INK, spaceAfter=7, wordWrap="CJK"),
    "small": ParagraphStyle("small", fontName="Korean", fontSize=7.6, leading=11.4, textColor=MUTED, spaceAfter=5, wordWrap="CJK"),
    "h1": ParagraphStyle("h1", fontName="KoreanBold", fontSize=20, leading=28, textColor=NAVY, spaceAfter=14, wordWrap="CJK"),
    "h2": ParagraphStyle("h2", fontName="KoreanBold", fontSize=11.5, leading=17, textColor=TEAL, spaceBefore=9, spaceAfter=7, wordWrap="CJK"),
    "cover": ParagraphStyle("cover", fontName="KoreanBold", fontSize=31, leading=44, textColor=NAVY, spaceAfter=16, wordWrap="CJK"),
    "tag": ParagraphStyle("tag", fontName="KoreanBold", fontSize=9, leading=14, textColor=TEAL, spaceAfter=8),
    "byline": ParagraphStyle("byline", fontName="KoreanBold", fontSize=11, leading=17, textColor=NAVY, spaceAfter=4, wordWrap="CJK"),
    "cell": ParagraphStyle("cell", fontName="Korean", fontSize=8, leading=11.5, textColor=INK, wordWrap="CJK"),
    "th": ParagraphStyle("th", fontName="KoreanBold", fontSize=8, leading=11.5, textColor=colors.white, wordWrap="CJK"),
    "metric": ParagraphStyle("metric", fontName="KoreanBold", fontSize=24, leading=31, textColor=TEAL, alignment=TA_CENTER),
    "metric_label": ParagraphStyle("metric_label", fontName="Korean", fontSize=8.3, leading=13, textColor=MUTED, alignment=TA_CENTER),
}


def p(text, style="body"):
    return Paragraph(text, ST[style])


def table(data, widths=None, small=False):
    cells = [[p(escape(str(c)), "th" if i == 0 else "cell") for c in row] for i, row in enumerate(data)]
    t = Table(cells, colWidths=widths or [WIDTH / len(data[0])] * len(data[0]), hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6F8")]),
        ("LINEBELOW", (0, 0), (-1, 0), 1, TEAL),
        ("LINEBELOW", (0, -1), (-1, -1), .5, colors.HexColor("#D6E0E5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5 if small else 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5 if small else 6),
    ]))
    return t


def fig(name, width=WIDTH):
    path = ROOT / "figures" / name
    with PILImage.open(path) as im:
        w, h = im.size
    return Image(str(path), width=width, height=width*h/w, hAlign="CENTER")


def callout(title, text):
    t = Table([[p(title, "h2")], [p(text)]], colWidths=[WIDTH])
    t.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,-1), PALE),
                           ("BOX",(0,0),(-1,-1),.5,colors.HexColor("#D3E8EC")),
                           ("LEFTPADDING",(0,0),(-1,-1),12),
                           ("RIGHTPADDING",(0,0),(-1,-1),12),
                           ("BOTTOMPADDING",(0,-1),(-1,-1),10)]))
    return t


HEADER_RIGHT = "실제 실행 결과"
FOOTER_LEFT = "특성 공학(Feature Engineering) 파이프라인 과제"


def decorate(c, doc):
    c.saveState()
    c.setFillColor(TEAL)
    c.rect(0, A4[1]-8, A4[0], 8, stroke=0, fill=1)
    c.setFont("Korean", 7)
    c.setFillColor(MUTED)
    c.drawString(44, A4[1]-33, "TITANIC / FEATURE ENGINEERING LAB")
    c.drawRightString(A4[0]-44, A4[1]-33, HEADER_RIGHT)
    c.setStrokeColor(colors.HexColor("#D7E1E7"))
    c.line(44, 36, A4[0]-44, 36)
    c.setFont("Korean", 7)
    c.drawString(44, 23, FOOTER_LEFT)
    c.restoreState()


class NumberedCanvas(pdfcanvas.Canvas):
    """전체 쪽수를 알고 난 뒤 '현재 / 전체' 쪽 번호를 그린다. 쪽수를 코드에 고정하지 않는다."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pages = []

    def showPage(self):
        self._pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._pages)
        for state in self._pages:
            self.__dict__.update(state)
            self.setFont("Korean", 7)
            self.setFillColor(MUTED)
            self.drawRightString(A4[0]-44, 23, f"{self._pageNumber:02d} / {total:02d}")
            super().showPage()
        super().save()


def build(name=None, student_id=None, department=None, course="빅데이터분석 01분반", github_url=None):
    global HEADER_RIGHT, FOOTER_LEFT
    s = json.loads((ROOT/"results/summary.json").read_text(encoding="utf-8"))
    run_date = (datetime.fromisoformat(s["executed_utc"]) + timedelta(hours=9)).date()  # 한국 시간 기준 실행일
    HEADER_RIGHT = f"실제 실행 결과 | {run_date:%Y.%m.%d}"
    FOOTER_LEFT = " · ".join(x for x in [course, "특성 공학(Feature Engineering) 파이프라인 과제"] if x)
    author = " ".join(x for x in [department, student_id, name] if x)
    folds = pd.read_csv(ROOT/"results/cv_fold_results.csv")
    shap_table = pd.read_csv(ROOT/"results/shap_importance.csv")
    cv = pd.read_csv(ROOT/"results/cv_results.csv")
    ab = pd.read_csv(ROOT/"results/controlled_comparisons.csv")
    ranks = pd.read_csv(ROOT/"results/feature_selection.csv")
    missing = pd.read_csv(ROOT/"results/data_dictionary.csv").set_index("column")
    corr = pd.read_csv(ROOT/"results/correlation_train.csv", index_col=0)
    def score(exp, model, metric="F1"):
        return float(cv.query("experiment == @exp and model == @model").iloc[0][metric])
    def f(x):
        return f"{x:.4f}"
    def train_f1(exp, model):
        return float(cv.query("experiment == @exp and model == @model").iloc[0]["F1_train"])
    def gap(exp, model):
        return float(cv.query("experiment == @exp and model == @model").iloc[0]["F1_gap"])
    def gap_change(model):
        """Top-10과 Ref의 (학습 F1 - 검증 F1) 격차를 fold별로 짝지어 뺀 값의 평균과 SD."""
        def per_fold(exp):
            d = folds[(folds.experiment == exp) & (folds.model == model)].set_index("fold")
            return d.F1_train - d.F1
        diff = per_fold("Top-10") - per_fold("Ref")
        return float(diff.mean()), float(diff.std(ddof=1))
    def fs(exp, model):
        r = cv.query("experiment == @exp and model == @model").iloc[0]
        return f"{r.F1:.4f} ± {r.F1_sd:.4f}"
    story = []
    def add(*items):
        story.extend(items)
    def page(title):
        if story:
            add(PageBreak())
        add(p(title, "h1"))

    # 1 / 표지 및 핵심 결과
    add(Spacer(1,27), p(escape(" / ".join(x for x in [course, "특성 공학 과제 보고서"] if x)), "tag"),
        p("타이타닉 생존 예측<br/>특성 공학 비교 실험", "cover"),
        p("전처리 선택이 모델에 미치는 영향을 실제 데이터로 확인하다", "h2"),
        p("Titanic 공개 데이터 891명 · 전처리 12개 설정 · 모델 2종 · 동일한 5-fold 교차검증"))
    if author:
        add(p(escape(author), "byline"))
    add(p(f'소스코드: <link href="{escape(github_url)}" color="#147D92">{escape(github_url)}</link>' if github_url
          else "소스코드: [GitHub 저장소 주소 입력 필요]", "small"), Spacer(1,12))
    cards = Table([
        [p(f"{s['holdout']['Accuracy']*100:.2f}%", "metric"), p(f(s["holdout"]["F1"]), "metric"), p(f(s["holdout"]["ROC_AUC"]), "metric")],
        [p("최종 평가 정확도", "metric_label"), p("최종 평가 F1", "metric_label"), p("최종 평가 ROC AUC", "metric_label")],
    ], colWidths=[WIDTH/3]*3)
    cards.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),PALE), ("TOPPADDING",(0,0),(-1,0),17),
                              ("BOTTOMPADDING",(0,-1),(-1,-1),17)]))
    add(cards, p("학습에 쓰지 않은 179명 평가 결과. 생존(1)을 양성으로 정의했으며, 판정 임계값은 0.5이다.", "small"), Spacer(1,16),
        callout("최종 선택 모델", "중앙값 대체 + One-Hot Encoding + StandardScaler + 파생 변수 4개 + 중요도 상위 10개 선택 + Random Forest. 교차검증으로 설정을 선택한 뒤 GridSearchCV를 적용했다."),
        p("이 실험에서 확인한 세 가지", "h2"),
        p(f"<b>1. 파생 변수의 효과는 모델마다 다르다.</b> 동일 조건에서 파생 변수 추가 후 LR의 CV F1은 {(score('Ref','LR')-score('No-FE','LR'))*100:.2f}%p 높아졌고, RF는 {(score('Ref','RF')-score('No-FE','RF'))*100:.2f}%p 높아졌다."),
        p(f"<b>2. 변수 선택이 항상 유리하지는 않다.</b> Top 10 선택은 RF에서 {(score('Top-10','RF')-score('Ref','RF'))*100:.2f}%p 개선됐지만 LR에서는 {(score('Ref','LR')-score('Top-10','LR'))*100:.2f}%p 낮아졌다."),
        p("<b>3. 높은 점수보다 비교 설계가 중요하다.</b> 전처리는 각 학습 fold 안에서 적합했다. 최종 평가 데이터는 설정 선택에 사용하지 않았다."),
        Spacer(1,8), p("보고서 구성: 데이터와 검증 설계 / EDA / 특성 공학 / 비교 실험 / 변수 선택과 과적합 / 최종 평가 / SHAP 설명 / 결론 / 재현 방법", "small"))

    # 2 / 데이터와 검증 설계
    page("01. 데이터와 검증 설계")
    add(p("Kaggle Titanic 생존 예측 데이터의 공개 CSV 미러를 사용했다 [1, 2]. 데이터 크기는 <b>(891, 12)</b>이며, 목표는 Survived(0=사망, 1=생존) 분류이다. 수치형과 범주형 변수가 함께 존재해 과제의 데이터 조건을 충족한다."))
    descriptions = [
        ("PassengerId","식별자","모델 입력 제외"),("Survived","생존 여부","타깃"),
        ("Pclass","객실 등급 1/2/3","범주형; Base만 원 숫자 사용"),("Name","승객 이름","Title 추출에만 사용"),
        ("Sex","성별","범주형 인코딩"),("Age","나이","수치형 결측치 비교"),
        ("SibSp","동승 형제자매/배우자 수","수치형; 가족 파생 변수"),
        ("Parch","동승 부모/자녀 수","수치형; 가족 파생 변수"),
        ("Ticket","티켓 번호","원 문자열 입력 제외"),("Fare","운임","수치형; 인당 운임 파생"),
        ("Cabin","객실 번호","높은 결측률로 입력 제외"),("Embarked","승선 항구 C/Q/S","최빈값 대체 후 인코딩"),
    ]
    data = [["컬럼","의미","전체 결측 수 (%)","사용 방식"]]
    for col, desc, use in descriptions:
        row = missing.loc[col]
        data.append([col,desc,f"{int(row.missing_all)} ({row.missing_all_pct:.2f}%)",use])
    add(table(data,[83,124,103,WIDTH-310],small=True), p("표 1. 원본 데이터의 변수 설명과 전체 891명 기준 결측 현황. 원본 중복 행 0개, PassengerId 중복 0개.","small"),
        p("데이터 분할과 선택 기준", "h2"),
        table([["구간","표본 수","사용 목적"],["학습 80%","712명 (사망 439 / 생존 273)","EDA, 5-fold CV, 모델·설정 선택"],
               ["최종 평가 20%","179명 (사망 110 / 생존 69)","선택 완료 후 최종 성능 산출"]],[85,191,WIDTH-276]),
        p("분할과 CV 모두 random_state=42, 계층화(stratify)를 적용했다. 5-fold의 검증 크기는 142~143명이다. 모든 실험은 동일 분할을 공유하며, <b>평균 CV F1</b>이 높은 설정을 우선 선택하고 동률이면 ROC AUC를 비교한다."),
        p("학습 누수 방지", "h2"),
        p("파생 변수 생성, 결측값 대체, 인코딩, 스케일링, 변수 선택, 모델을 Pipeline으로 연결했다 [3, 4]. 대체값·인코딩 사전·스케일 통계·변수 중요도는 각 학습 fold에서만 학습한다. 이후 EDA 그림과 이상치 탐지는 학습 712명 기준이다."))

    # 3 / 데이터 품질과 분포
    page("02. EDA: 결측값과 이상치")
    add(fig("05_missingness.png", WIDTH-30),
        p("Cabin은 학습 표본의 77.53%가 비어 있어 기본 입력에서 제외했다. Age 137개는 대체 전략을 비교하고, Embarked 2개는 범주형 최빈값으로 처리했다. 원본 전체 결측 수와 학습 결측 수는 구분했다."),
        fig("01_histograms.png"),
        p("그림 1. 필수 Histogram. 나이는 결측값을 제외해 그렸고, 운임은 상위 일부 값이 큰 오른쪽 비대칭 분포를 보인다.","small"),
        p("IQR 기준 이상치 탐지", "h2"))
    out = [["변수","하한 / 상한","탐지 수 / 학습 712명","판단"]]
    notes = {"Age":"고령 승객 가능","Fare":"고가 운임 가능","SibSp":"큰 가족 가능","Parch":"IQR=0; 기계적 제거 부적절"}
    for row in s["eda"]["outliers"]:
        out.append([row["feature"],f"{row['lower']:.2f} / {row['upper']:.2f}",f"{row['outliers']} ({row['outlier_pct']:.2f}%)",notes[row["feature"]]])
    add(table(out,[65,118,140,WIDTH-323],small=True),
        p("기준: Q1 - 1.5×IQR 미만 또는 Q3 + 1.5×IQR 초과. 비율의 분모는 학습 712명이며 결측 Age는 탐지 수에 포함하지 않았다. Parch는 Q1=Q3=0이므로 0보다 큰 값이 모두 표시된다. 탐지값을 오류로 단정하지 않고 유지했으며, RobustScaler도 비교했다.","small"))

    # 4 / 타깃과 관계
    page("03. EDA: 타깃 분포와 변수 관계")
    add(fig("04_target_and_groups.png"),
        p("학습 생존 비율은 273/712=38.34%로 다소 불균형하다. 정확도 외에 Precision·Recall·F1·ROC AUC를 함께 보고한다. 여성/남성 생존 비율은 74.31%/18.52%, 1/2/3등급은 64.91%/44.67%/24.30%였다."),
        fig("02_boxplots.png", WIDTH-25),
        p("그림 2. 필수 Countplot, Barplot, Boxplot. 운임의 극단값을 생략하지 않고 원 단위로 표시했다.","small"),
        fig("03_heatmap.png",340),
        p(f"그림 3. 필수 Heatmap. Survived와 Pclass의 상관은 {corr.loc['Survived','Pclass']:.2f}, Fare는 {corr.loc['Survived','Fare']:.2f}였다. Pearson 상관은 수치형 변수의 선형 관계만 보여 주며, 결측치는 쌍별로 제외한다. 성별·등급의 관측 차이를 인과효과로 해석하지 않는다.","small"))

    # 5 / 파이프라인
    page("04. 특성 공학과 실험 구성")
    add(p("원본 입력 → 행 단위 파생 변수 → ColumnTransformer → 중요도 선택 → 분류 모델", "h2"),
        table([["단계","구현 / 비교"],
               ["결측값", "수치형 Mean / Median / Most Frequent 3개. 범주형은 모든 실험에서 Most Frequent로 고정."],
               ["인코딩", "OneHotEncoder와 OrdinalEncoder 비교. 과제의 Label Encoding 방식은 입력 변수별 정수 매핑으로 구현; 타깃용 LabelEncoder를 특성에 사용하지 않음."],
               ["스케일링", "Standard / MinMax / Robust + 미적용. 수치형에만 적용; 범주 인코딩 결과는 그대로 사용."],
               ["변수 선택", "SelectFromModel(RandomForest): 불순도 중요도 상위 10개. 선택 RF는 120 trees, depth=6, min_samples_leaf=2."],
               ["학습 모델", "LR: C=1, lbfgs, max_iter=5000. RF: 180 trees, depth=6, min_samples_leaf=2. class_weight=None, seed=42."],
              ],[80,WIDTH-80]),
        p("필수 파생 변수: 총 4개", "h2"),
        table([["변수","공식 / 고정 규칙","의도"],
               ["FamilySize", "SibSp + Parch + 1", "동승 가족 규모"],
               ["FarePerPerson", "Fare / FamilySize", "가족 수로 나눈 운임 대리 변수"],
               ["IsAlone", "FamilySize == 1이면 1", "단독 승객 여부"],
               ["Title", "Name에서 호칭 추출; Mr/Mrs/Miss/Master/Rare", "이름의 범주 정보 요약"],
              ],[93,240,WIDTH-333],small=True),
        p("Mlle/Ms는 Miss, Mme는 Mrs로 통합하고 나머지는 Rare로 묶는 고정 규칙을 사용했다. FarePerPerson은 티켓별 실제 결제 인원과 다를 수 있는 대리 변수이다. 원본 Name은 모델에 직접 입력하지 않는다.","small"),
        p("과제에서 요구한 네 조합", "h2"),
        table([["실험","수치 결측값","인코딩","스케일링","변수 선택"],
               ["Base","없음","없음","없음","없음"],
               ["Exp-1","Mean","One-Hot","Standard","X"],
               ["Exp-2","Median","정수 매핑","MinMax","O (Top 10)"],
               ["Exp-3","Most Frequent","One-Hot","Robust","O (Top 10)"],
              ],[55,99,100,105,WIDTH-359],small=True),
        p("Base는 변환 없이 모델을 실행하기 위해 결측이 없는 Pclass·SibSp·Parch·Fare만 사용한다. 행 삭제는 없으며 동일 표본을 유지한다. 다른 실험과는 정보량도 다르므로 Base 대비 차이를 순수한 전처리 효과로 해석하지 않는다. Base를 제외한 필수 조합에는 파생 변수 4개가 공통 적용된다.","small"))

    # 6 / 필수 비교 성능
    page("05. 필수 조합의 모델 성능 비교")
    add(p("다음 표는 최종 평가 데이터가 아닌, 학습 데이터의 동일한 5-fold 교차검증 평균이다. 양성은 생존(1)이며, F1 외의 네 지표도 실제 계산했다."))
    rows = [["실험","모델","Accuracy","Precision","Recall","F1","ROC AUC"]]
    for exp in ["Base","Exp-1","Exp-2","Exp-3"]:
        for model in ["LR","RF"]:
            r = cv.query("experiment == @exp and model == @model").iloc[0]
            rows.append([exp,model]+[f(r[m]) for m in ["Accuracy","Precision","Recall","F1","ROC_AUC"]])
    add(table(rows,[58,40,82,82,72,75,WIDTH-409],small=True),
        p("LR = Logistic Regression, RF = Random Forest. 평균은 fold별 점수의 산술평균이다. 정밀도/재현율을 먼저 평균한 뒤 F1을 계산한 값과는 다를 수 있다.","small"),
        p("필수 조합 + 통제 비교 전체 24개 실험", "h2"), fig("06_cv_comparison.png"),
        p("오차 막대는 5개 fold 점수의 표본 표준편차(SD)이며 신뢰구간이 아니다. 두 모델에 같은 전처리를 적용해도 결과가 같지 않다.","small"),
        callout("복합 조합의 승자를 원인으로 오해하지 않기", "필수 조합 중 RF의 Exp-3 F1이 가장 높다. 그러나 결측 처리·스케일링·변수 선택이 함께 바뀌었으므로 어느 단계가 원인인지 이 표만으로 결론낼 수 없다. 다음 페이지는 Ref를 기준으로 한 번에 한 설정만 바꾼 비교이다."))

    # 7 / 통제 비교
    page("06. 한 요소씩 바꾼 통제 비교")
    add(p("Ref는 Median + One-Hot + Standard + 파생 변수 4개 + 변수 선택 없음이다. 아래 설정은 별도 표시가 없으면 Ref와 동일하다."))
    definitions = {"Ref":"비교 기준", "Exp-1":"결측값 → Mean", "Mode":"결측값 → Most Frequent",
                   "Ordinal":"인코딩 → 정수 매핑", "MinMax":"스케일 → MinMax", "Robust":"스케일 → Robust",
                   "No-Scale":"스케일링 미적용", "No-FE":"파생 변수 4개 미사용", "Top-10":"상위 10개 변수 선택"}
    data=[["실험 / 변경 요소","LR CV F1 ± SD","RF CV F1 ± SD"]]
    for exp, desc in definitions.items():
        data.append([f"{exp}: {desc}",fs(exp,"LR"),fs(exp,"RF")])
    add(table(data,[215,146,WIDTH-361],small=True),
        p("동일 fold에서 계산한 F1 변화량", "h2"),
        p("아래 값은 변경 설정 - Ref이다. 양수면 개선, 음수면 저하이며 단위는 %p이다. 예를 들어 +3.10%p는 F1이 0.0310 높아졌다는 뜻이다.","small"))
    labels = {"Mean vs Median":"평균 대체", "Mode vs Median":"최빈값 대체", "Ordinal vs OneHot":"정수 매핑 인코딩",
              "MinMax vs Standard":"MinMax 스케일", "Robust vs Standard":"Robust 스케일", "No scale vs Standard":"스케일링 미적용",
              "Top10 vs All":"Top 10 선택", "No FE vs FE":"파생 변수 제거"}
    data=[["변경 요소","LR 변화 ± SD (%p)","RF 변화 ± SD (%p)"]]
    for label, korean in labels.items():
        row=[korean]
        for model in ["LR","RF"]:
            r=ab.query("comparison == @label and model == @model").iloc[0]
            row.append(f"{r.delta_f1*100:+.2f} ± {r.delta_sd*100:.2f}")
        data.append(row)
    add(table(data,[215,146,WIDTH-361],small=True),
        p("차이의 SD는 fold별 대응 차이로 계산했다. 5개 fold는 서로 독립적인 반복 실험이 아니며, 이 표로 통계적 유의성을 주장하지 않는다. 전체 24개 실험의 모든 지표와 fold별 원점수는 CSV로 함께 제공한다.","small"))

    # 8 / 변수 선택과 설명
    page("07. 변수 선택과 중요도 해석")
    add(p("중요도 기반 선택은 학습 fold 안에서 수행했다. 아래 목록은 설명을 위해 학습 712명 전체로 다시 적합한 Top-10 파이프라인의 결과이다. 각 CV fold의 선택 변수는 달라질 수 있다."),
        fig("08_feature_importance.png", WIDTH-90),
        p("그림 4. 왼쪽은 변수 선택 RF의 불순도 중요도, 오른쪽은 최종 모델의 평가 데이터 순열 중요도이다. 두 중요도는 계산 기준과 변수가 다르므로 막대 크기를 직접 비교하지 않는다.","small"),
        table([["순위","선택 변수","중요도","순위","선택 변수","중요도"]] +
              [[str(i+1),ranks.iloc[i].feature,f"{ranks.iloc[i].importance:.4f}",str(i+6),ranks.iloc[i+5].feature,f"{ranks.iloc[i+5].importance:.4f}"] for i in range(5)],
              [30,139,60,30,139,WIDTH-398],small=True),
        p("제거 전후 비교: 성능과 과적합", "h2"),
        p("One-Hot 변환 후 <b>20개 특성에서 10개</b>로 축소했다. 같은 5개 fold에서 학습 부분과 검증 부분의 F1을 함께 계산해, 성능 변화와 과적합(학습 - 검증 격차) 변화를 같이 본다."),
        table([["설정","모델","학습 F1","검증 F1","격차 (학습 - 검증)"]] +
              [[label, model, f"{train_f1(exp,model):.4f}", f"{score(exp,model):.4f}", f"{gap(exp,model)*100:+.2f}%p"]
               for model in ["RF","LR"] for exp,label in [("Ref","전체 20개"),("Top-10","상위 10개")]],
              [100,60,100,100,WIDTH-360],small=True),
        p(f"RF는 변수 선택 후 검증 F1이 오르고 격차가 {gap('Ref','RF')*100:.2f}%p → {gap('Top-10','RF')*100:.2f}%p로 {abs(gap_change('RF')[0])*100:.2f}%p 줄었다. "
          f"LR은 검증 F1이 낮아졌고 격차는 {gap('Ref','LR')*100:.2f}%p → {gap('Top-10','LR')*100:.2f}%p로 변화가 없었다. "
          f"격차 변화의 fold별 대응 차이 SD는 RF {gap_change('RF')[1]*100:.2f}%p, LR {gap_change('LR')[1]*100:.2f}%p이다. RF 중요도로 고른 변수 집합이 LR에 최적이라는 보장은 없다.","small"),
        p("중요도의 한계", "h2"),
        p("불순도 중요도는 연속형/고유값이 많은 변수에 유리할 수 있고, 관련 변수 사이에서 중요도가 분산될 수 있다. Sex의 두 One-Hot 열은 중복 정보를 포함한다. 상위 10개 선택이 최소한의 독립적인 변수 집합을 의미하지는 않는다.","small"),
        p("순열 중요도는 원본 컬럼을 하나씩 섞어 ROC AUC 감소량을 30회 측정한 값이다 [5]. Name은 Title의 입력이므로 이름 자체를 외운 효과로 해석하면 안 된다. Fare·SibSp·Parch를 섞으면 관련 파생 변수도 함께 바뀐다. 평가 후 설명에만 사용했고 모델 재선택에는 사용하지 않았다.","small"))

    # 9 / 최종 평가
    page("08. 모델 선택과 최종 평가")
    add(p(f"전체 실험에서 CV F1이 가장 높은 <b>{s['selected_experiment']} / Random Forest</b>를 선택했다. 기본 설정의 CV F1은 {s['untuned_cv_f1']:.4f}였다. 이후 같은 학습 데이터의 CV로 12개 하이퍼파라미터 조합을 탐색했다."),
        table([["항목","내용"],["GridSearchCV 후보","max_depth=[4,6,8,None] × min_samples_leaf=[1,2,4]"],
               ["선택 파라미터","max_depth=8, min_samples_leaf=2; n_estimators=180 고정"],
               ["선택 후 CV F1",f"{s['tuned_cv_f1']:.4f}; 712명 전체로 refit='F1' 기준 재학습"],
               ["최종 판정", "predict() / 생존 확률 약 0.5 기준; 임계값 별도 최적화 없음"]],[105,WIDTH-105],small=True),
        p("최종 평가 179명: 선택 완료 후 측정", "h2"))
    data=[["지표","최종 점수","95% bootstrap 구간"]]
    for m in ["Accuracy","Precision","Recall","F1","ROC_AUC"]:
        lo,hi=s["holdout_ci95"][m]
        data.append([m,f(s["holdout"][m]),f"{lo:.4f} ~ {hi:.4f}"])
    add(table(data,[125,125,WIDTH-250],small=True), fig("07_holdout_evaluation.png"),
        p("사망 110명 중 98명, 생존 69명 중 50명을 맞혔다. 생존자를 사망으로 예측한 19명(FN)이 있어 재현율 개선 여지가 있다. F1은 100/(100+12+19)=0.7634이다."),
        p("95% 구간은 평가 표본에서 2,000회 행 단위 bootstrap의 2.5/97.5 분위수다. 모델 재학습/모델 선택/가족 간 의존성의 불확실성은 포함하지 않는다. CV에서 여러 설정을 선택했으므로 CV 점수는 낙관적일 수 있으며, 일반화 성능의 기준은 이 별도 평가 결과이다.","small"))

    # 10 / SHAP 설명
    page("09. 최종 모델 설명: SHAP")
    top = shap_table.iloc[0]
    shap_rank = {name: i+1 for i, name in enumerate(shap_table.feature)}
    derived = shap_table[shap_table.feature.isin(["FarePerPerson","FamilySize","IsAlone"]) | shap_table.feature.str.startswith("Title_")]
    add(p(f"SHAP은 한 승객의 예측을 '기준값 + 변수별 기여도의 합'으로 분해한다 [6]. 최종 Random Forest에 TreeExplainer를 적용해, 모델이 실제로 입력받는 "
          f"<b>선택된 10개 변수</b>가 평가 데이터 179명의 생존(1) 예측 확률에 준 기여를 계산했다. 기준값(평균 예측 확률)은 {s['shap']['base_value']:.4f}이며, "
          f"모든 승객에서 기준값 + 기여도 합이 모델의 예측 확률과 일치함을 확인했다(최대 오차 {s['shap']['additivity_max_error']:.1e})."),
        fig("09_shap_summary.png", WIDTH-110),
        p("그림 5. SHAP 요약. 점 하나가 승객 한 명이다. 가로 위치는 그 변수가 생존 확률을 올린(+) 또는 내린(-) 크기, 색은 변수 값의 높고 낮음이다. 수치형 변수는 표준화된 값 기준이다.","small"),
        table([["순위","변수","평균 |SHAP|","관측된 기여 경향","순위","변수","평균 |SHAP|","관측된 기여 경향"]] +
              [[str(i+1), shap_table.iloc[i].feature, f"{shap_table.iloc[i].mean_abs_shap:.4f}", "높은 값: 양(+)" if shap_table.iloc[i].value_shap_corr > 0 else "높은 값: 음(-)",
                str(i+6), shap_table.iloc[i+5].feature, f"{shap_table.iloc[i+5].mean_abs_shap:.4f}", "높은 값: 양(+)" if shap_table.iloc[i+5].value_shap_corr > 0 else "높은 값: 음(-)"]
               for i in range(5)],
              [36,84,58,84,36,76,58,WIDTH-432],small=True),
        p("'관측된 기여 경향'은 평가 표본 179명에서 변수 값과 SHAP 값의 상관 부호이다. 표본에서 관측된 경향이며, 값이 커질 때 기여가 항상 같은 방향으로 변한다는 뜻은 아니다. One-Hot 열은 해당 범주이면 1이다.","small"),
        p("해석", "h2"),
        p(f"<b>성별 정보가 예측을 주도한다.</b> 평균 |SHAP| 1위는 {top.feature}({top.mean_abs_shap:.4f})이고 Sex_male, Sex_female이 상위에 있다. "
          "평가 표본에서 호칭이 Mr이거나 남성인 승객은 생존 예측에 음의 기여를, 여성인 승객은 양의 기여를 받는 경향을 보였다. EDA에서 본 성별 생존 비율 차이(여성 74.31% / 남성 18.52%)와 같은 방향이다."),
        p(f"<b>파생 변수가 실제 예측에 쓰인다.</b> 선택된 10개 중 {len(derived)}개가 파생 변수(또는 그 One-Hot 열)이며, "
          f"FarePerPerson은 {shap_rank['FarePerPerson']}위로 원본 Fare({shap_rank['Fare']}위)보다 기여가 컸다. "
          "평가 표본에서 높은 1인당 운임은 생존 예측에 양의 기여를, Pclass_3·높은 Age·큰 FamilySize는 음의 기여를 하는 경향을 보였다."),
        p("<b>불순도 중요도와의 비교.</b> 07절의 불순도 중요도에서도 Title_Mr, Sex, FarePerPerson, Fare가 상위였다. 계산 방식이 다른 두 지표가 같은 변수를 가리키므로 변수 선택 결과를 뒷받침한다."),
        p("SHAP은 모델이 변수를 어떻게 사용했는지를 설명할 뿐 인과관계를 뜻하지 않는다. Title_Mr·Sex_male·Sex_female은 겹치는 정보를 담고 있어 기여가 세 열에 나뉘어 나타난다. 평가 완료 후 설명에만 사용했고 모델 선택에는 쓰지 않았다.","small"))

    # 11 / 결론
    page("10. 결과 해석과 최종 결론")
    answers = [
        ("1. 어떤 전처리가 가장 효과적이었는가?",
         f"단일 처리만의 보편적 우열은 확인할 수 없다. 이 실험의 RF에서는 Top 10 선택이 Ref 대비 F1을 {(score('Top-10','RF')-score('Ref','RF'))*100:.2f}%p 높여 가장 큰 긍정적 변화를 보였다. 결측값 대체만 바꾸면 최빈값의 CV F1이 LR {score('Mode','LR'):.4f}, RF {score('Mode','RF'):.4f}로 가장 높았지만 중앙값 대비 차이는 각각 0.10%p, 0.50%p로 작았다."),
        ("2. One-Hot Encoding이 항상 좋은가?",
         f"아니다. 같은 설정에서 LR은 One-Hot {score('Ref','LR'):.4f}, 정수 매핑 {score('Ordinal','LR'):.4f}로 One-Hot이 좋았다. RF는 {score('Ref','RF'):.4f}와 {score('Ordinal','RF'):.4f}로 정수 매핑이 소폭 높았다. 명목형 범주에 임의의 순서가 생기는 문제와 모델 특성을 함께 고려해야 한다."),
        ("3. Feature Selection이 과적합 감소에 기여했는가?",
         f"<b>RF에서만 부분적으로 기여했다.</b> 과적합의 지표로 학습 F1과 검증 F1의 격차를 비교했다(07절 표). RF는 Top 10 선택 후 격차가 {gap('Ref','RF')*100:.2f}%p → {gap('Top-10','RF')*100:.2f}%p로 줄고 "
         f"검증 F1도 {score('Ref','RF'):.4f} → {score('Top-10','RF'):.4f}로 올랐다. 다만 감소 폭 {abs(gap_change('RF')[0])*100:.2f}%p는 fold 간 변동(SD {gap_change('RF')[1]*100:.2f}%p)"
         f"{'보다 작아' if abs(gap_change('RF')[0]) < gap_change('RF')[1] else '보다 크지만 fold가 5개뿐이어서'} 강한 근거는 아니며, 선택 후에도 {gap('Top-10','RF')*100:.2f}%p의 격차가 남는다. "
         f"LR은 격차가 처음부터 {gap('Ref','LR')*100:.2f}%p로 작아 줄일 과적합이 거의 없었고, 선택 후 검증 F1은 오히려 {score('Ref','LR'):.4f} → {score('Top-10','LR'):.4f}로 낮아졌다. "
         "즉 변수 선택은 과적합 여지가 큰 모델에서만 도움이 됐다."),
        ("4. 스케일링은 모델에 어떤 영향을 주었는가?",
         "LR의 Standard/Robust/미적용 F1은 모두 0.7633이었고 MinMax는 0.7612였다. RF도 차이가 작았다. 트리의 분할 순서는 단조 변환에 대체로 유지되지만 구현의 수치 정밀도 등으로 작은 차이는 생길 수 있다. 이번 결과는 '스케일링이 일반적으로 불필요하다'는 근거가 아니다."),
        ("5. Feature Engineering은 실제로 얼마나 기여했는가?",
         f"같은 결측 처리·인코딩·스케일링·모델에서 파생 변수 4개를 추가했을 때 LR F1은 {score('No-FE','LR'):.4f} → {score('Ref','LR'):.4f} (+3.31%p), RF는 {score('No-FE','RF'):.4f} → {score('Ref','RF'):.4f} (+0.45%p)였다. 이는 4개 묶음의 효과이고, 개별 파생 변수의 효과를 분리한 실험은 아니다."),
    ]
    for title,text in answers:
        add(p(title,"h2"),p(text))
    add(callout("최종 결론", "이 데이터와 검증 설정에서는 파생 변수와 RF 기반 Top 10 선택을 적용한 Random Forest가 CV F1 기준으로 선택되었다. 최종 평가 정확도는 82.68%, F1은 0.7634였다. 성능을 높이는 처리는 모델에 따라 달랐으므로, Pipeline 안에서 공정하게 비교하고 검증 결과로 결정해야 한다."),
        p("남은 한계", "h2"),
        p("단일 랜덤 분할과 작은 평가 표본에 의존한다. 같은 가족/티켓 승객이 분할 양쪽에 포함될 수 있으므로 독립성 가정이 약해질 수 있다. 후속 연구로 그룹 분할, 반복/중첩 CV, 각 파생 변수의 개별 비교가 필요하다. Kaggle 비공개 테스트 순위나 실제 서비스 성능을 의미하지 않는다.","small"))

    # 12 / 재현성과 요구사항
    page("11. 재현 방법과 과제 조건 확인")
    add(p(f"Python {s['python']} 환경에서 실행했다. 제공된 폴더를 작업 경로로 열어 아래 명령을 실행한다. CSV를 동봉했으므로 분석 자체는 오프라인으로 재실행할 수 있다."),
        callout("실행 명령", "python -m pip install -r requirements.txt<br/>python analysis.py<br/>python build_report.py"),
        p("실행 노트북은 실제 저장 결과를 읽어 표와 그래프를 표시한다. RECOMPUTE=True로 바꾸면 전체 실험을 다시 수행한다. 데이터 SHA-256, 분할 PassengerId, 패키지 버전, fold별 점수와 최종 예측도 저장했다."),
        p("조건 충족 내역", "h2"),
        table([["과제 요구사항","구현 내용 / 확인 위치"],
               ["실제 지도학습 / 500개 이상 / 혼합형","Titanic 891명 / 데이터 소개와 변수 설명 (2쪽)"],
               ["EDA와 네 종류 필수 그래프","결측·이상치·분포·상관·타깃 분석 (3~4쪽)"],
               ["전처리별 2개 이상 비교","결측 3개 / 인코딩 2개 / 스케일 3개+미적용 (5~7쪽)"],
               ["파생 변수 최소 2개","FamilySize, FarePerPerson, IsAlone, Title (5쪽)"],
               ["상위 N개 선택 / 제거 전후 비교","RF 중요도 Top 10 / 전체 변수와 성능·과적합 격차 비교 (7~8쪽)"],
               ["모델 최소 2개 / 필수 조합 표","LR + RF / Base와 Exp-1~3 (6쪽)"],
               ["평가와 최종 결론","5개 분류 지표, 혼동행렬, ROC (9쪽) / 질문별 해석 (11쪽)"],
               ["추가 항목 구현","Pipeline / GridSearchCV (9쪽) / SHAP 설명 (10쪽) / 중요도 시각화 (8쪽). LIME·AutoML 미수행"],
              ],[176,WIDTH-176],small=True),
        p("핵심 실행 환경", "h2"),
        p(" · ".join(f"{name} {s['versions'][name]}" for name in ["pandas","numpy","scikit-learn","shap","matplotlib","seaborn"]),"small"),
        p("참고 자료", "h2"))
    sources=[
        ("1", "Kaggle, Titanic - Machine Learning from Disaster", "https://www.kaggle.com/competitions/titanic/data"),
        ("2", "Data Science Dojo, Titanic CSV 공개 미러 (실제 사용 데이터)", s["data_url"]),
        ("3", "scikit-learn, Column Transformer with Mixed Types", "https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html"),
        ("4", "scikit-learn, Common pitfalls and recommended practices", "https://scikit-learn.org/stable/common_pitfalls.html"),
        ("5", "scikit-learn, Permutation feature importance", "https://scikit-learn.org/stable/modules/permutation_importance.html"),
        ("6", "SHAP documentation (Lundberg & Lee, A Unified Approach to Interpreting Model Predictions, 2017)", "https://shap.readthedocs.io/en/latest/"),
    ]
    for number,title,url in sources:
        add(p(f'[{number}] <link href="{escape(url)}" color="#147D92">{escape(title)}</link>',"small"))
    add(p(f"출처 확인 및 실험 실행: {run_date:%Y-%m-%d}. 패키지 버전과 원본 해시는 results/summary.json 및 requirements.txt에 기록했다.","small"),
        p(f'소스코드와 노트북: <link href="{escape(github_url)}" color="#147D92">{escape(github_url)}</link>' if github_url
          else "소스코드와 노트북: [GitHub 저장소 주소 입력 필요]","small"))
    output=ROOT/"Titanic_Feature_Engineering_Report.pdf"
    doc=SimpleDocTemplate(str(output),pagesize=A4,leftMargin=44,rightMargin=44,topMargin=55,bottomMargin=48,
                          title="타이타닉 생존 예측: 특성 공학 비교 실험",author=author or "",pageCompression=1)
    doc.build(story,onFirstPage=decorate,onLaterPages=decorate,canvasmaker=NumberedCanvas)
    print(output)
    return output


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", help="표지에 넣을 이름")
    parser.add_argument("--student-id", help="표지에 넣을 학번")
    parser.add_argument("--department", help="표지에 넣을 학과")
    parser.add_argument("--course", default="빅데이터분석 01분반", help="과목·분반")
    parser.add_argument("--github-url", help="소스코드 GitHub 저장소 주소")
    args = parser.parse_args()
    build(args.name, args.student_id, args.department, args.course, args.github_url)
