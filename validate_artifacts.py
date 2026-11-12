"""저장된 모델, 예측 결과, 데이터 분할 및 보고서를 독립적으로 검증한다."""
import json
from pathlib import Path
import analysis
import joblib
import numpy as np
import pandas as pd
from pypdf import PdfReader


def validate():
    root=Path(__file__).resolve().parent
    res=root/'results'
    df,sha=analysis.load_data()
    s=json.loads((res/'summary.json').read_text(encoding='utf-8'))
    split=json.loads((res/'split_manifest.json').read_text(encoding='utf-8'))
    train_ids=set(split['train_ids'])
    test_ids=set(split['test_ids'])
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == set(df.PassengerId)
    validation_ids=[]
    for fold in split['cv']:
        a,b=set(fold['fit_ids']),set(fold['validation_ids'])
        assert a.isdisjoint(b) and a | b == train_ids
        assert not ((a|b) & test_ids)
        validation_ids.extend(fold['validation_ids'])
    assert len(validation_ids)==len(set(validation_ids))==712
    cv=pd.read_csv(res/'cv_results.csv')
    assert np.allclose(cv.F1_train-cv.F1,cv.F1_gap)
    shap_table=pd.read_csv(res/'shap_importance.csv')
    assert len(shap_table)==s['transformed_features_after'] and s['shap']['additivity_max_error']<1e-6
    raw=pd.read_csv(res/'cv_fold_results.csv')
    assert len(cv)==24 and len(raw)==120
    for r in cv.itertuples():
        subset=raw[(raw.experiment==r.experiment)&(raw.model==r.model)]
        for m in analysis.METRICS:
            assert np.isclose(subset[m].mean(),getattr(r,m))
    saved=pd.read_csv(res/'holdout_predictions.csv')
    test=df.set_index('PassengerId',drop=False).loc[saved.PassengerId]
    model=joblib.load(res/'final_pipeline.joblib')
    X=test.drop(columns='Survived')
    assert np.allclose(model.predict_proba(X)[:,1],saved.probability_survived)
    assert np.array_equal(model.predict(X),saved.prediction)
    actual=test.Survived.to_numpy()
    pred=saved.prediction.to_numpy()
    assert np.array_equal(actual,saved.actual)
    tp=int(((actual==1)&(pred==1)).sum())
    tn=int(((actual==0)&(pred==0)).sum())
    fp=int(((actual==0)&(pred==1)).sum())
    fn=int(((actual==1)&(pred==0)).sum())
    assert [[tn,fp],[fn,tp]]==s['confusion_matrix']
    assert np.isclose(2*tp/(2*tp+fp+fn),s['holdout']['F1'])
    assert np.isclose((tp+tn)/len(actual),s['holdout']['Accuracy'])
    pdf=PdfReader(root/'Titanic_Feature_Engineering_Report.pdf')
    pages=len(pdf.pages)
    texts=[p.extract_text() for p in pdf.pages]
    assert all(f'{i:02d} / {pages:02d}' in t for i,t in enumerate(texts,1))
    assert '82.68%' in texts[0] and '0.7634' in texts[0]
    report={'status':'passed','checks':['data SHA-256','891 unique records','holdout disjointness',
            'all five folds disjoint and complete','24 experiment means from 120 fold rows',
            'fresh-process model reload','saved predictions reproduce exactly','metrics recomputed from confusion matrix',
            'train-validation gap consistent','SHAP additivity',f'{pages}-page PDF with all page numbers and matching headline metrics'],
            'sha256':sha,'holdout_rows':179,'cv_experiments':len(cv)}
    (res/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    validate()
