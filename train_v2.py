import argparse, os, json
import pandas as pd, xgboost as xgb, joblib
from sklearn.metrics import precision_recall_curve, auc, f1_score

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n_estimators', type=int, default=300)
    p.add_argument('--max_depth', type=int, default=5)
    p.add_argument('--eta', type=float, default=0.1)
    p.add_argument('--subsample', type=float, default=0.9)
    p.add_argument('--min_child_weight', type=int, default=1)
    p.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    p.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    p.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    a = p.parse_args()

    tr = pd.read_csv(os.path.join(a.train, 'train.csv'), header=None)
    va = pd.read_csv(os.path.join(a.validation, 'validation.csv'), header=None)
    X_tr, y_tr = tr.iloc[:, 1:], tr.iloc[:, 0]
    X_va, y_va = va.iloc[:, 1:], va.iloc[:, 0]
    spw = (y_tr == 0).sum() / (y_tr == 1).sum()
    print("scale_pos_weight:", round(float(spw), 2))

    m = xgb.XGBClassifier(n_estimators=a.n_estimators, max_depth=a.max_depth,
        learning_rate=a.eta, subsample=a.subsample, min_child_weight=a.min_child_weight,
        scale_pos_weight=spw, eval_metric='aucpr', tree_method='hist', n_jobs=-1, random_state=42)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=30, verbose=False)

    pr = m.predict_proba(X_va)[:, 1]
    prec, rec, _ = precision_recall_curve(y_va, pr)
    pr_auc = auc(rec, prec); f1 = f1_score(y_va, (pr >= 0.5).astype(int))
    print(f"val_pr_auc: {pr_auc:.6f}")
    print(f"val_f1: {f1:.6f}")
    joblib.dump(m, os.path.join(a.model_dir, 'model.joblib'))
    json.dump({'val_pr_auc': float(pr_auc), 'val_f1': float(f1)},
              open(os.path.join(a.model_dir, 'metrics.json'), 'w'))
