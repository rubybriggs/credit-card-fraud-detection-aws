import os, sys, subprocess, tarfile, json
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xgboost==1.7.6', '-q'])
import pandas as pd, numpy as np, joblib
from sklearn.metrics import precision_recall_curve, auc, f1_score

if __name__ == '__main__':
    with tarfile.open('/opt/ml/processing/model/model.tar.gz') as t:
        t.extractall('/opt/ml/processing/model/x')
    m = joblib.load('/opt/ml/processing/model/x/model.joblib')
    te = pd.read_csv('/opt/ml/processing/test/test.csv', header=None)
    y = te.iloc[:, 0].values; X = te.iloc[:, 1:]
    assert set(np.unique(y)) <= {0, 1}, np.unique(y)[:5]
    pr = m.predict_proba(X)[:, 1]
    prec, rec, _ = precision_recall_curve(y, pr)
    top = np.argsort(pr)[::-1][:int(len(y)*0.05)]
    rep = {"metrics": {"pr_auc": {"value": float(auc(rec, prec))},
                       "f1": {"value": float(f1_score(y, (pr >= 0.5).astype(int)))},
                       "recall_at_5pct": {"value": float(y[top].sum()/y.sum())}}}
    os.makedirs('/opt/ml/processing/output/evaluation', exist_ok=True)
    json.dump(rep, open('/opt/ml/processing/output/evaluation/evaluation.json', 'w'), indent=2)
    print("EVALUATION:", json.dumps(rep))
