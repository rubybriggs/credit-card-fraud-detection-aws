import os, io, json, logging
import numpy as np, pandas as pd, joblib
logger = logging.getLogger(__name__)
model = None; scaler = None

def model_fn(model_dir):
    global model, scaler
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    sp = os.path.join(model_dir, 'preprocessor.joblib')
    if os.path.exists(sp):
        scaler = joblib.load(sp)['scaler']
    try:
        model.set_params(n_jobs=1)      # single-thread scoring: kills thread thrash under concurrency
    except Exception:
        model.n_jobs = 1
    logger.info("model+scaler loaded (single-thread scoring)")
    return model

def input_fn(body, content_type='text/csv'):
    if isinstance(body, (bytes, bytearray)): body = body.decode('utf-8')
    if content_type == 'application/json':
        return pd.DataFrame([json.loads(body)])
    try:    return pd.read_csv(io.StringIO(body))
    except Exception: return pd.DataFrame([json.loads(body)])

def predict_fn(df, _mdl):
    df = df.copy(); df.columns = [str(c).lower() for c in df.columns]
    if 'class' not in df.columns: df['class'] = 0
    if 'txn_sequence' in df.columns: df = df.sort_values('txn_sequence').reset_index(drop=True)
    df['hour_sin'] = np.sin(2*np.pi*df['hour_of_day']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour_of_day']/24)
    if scaler is not None and 'amount' in df.columns:
        df['amount_scaled'] = scaler.transform(df[['amount']])
    for c in ['time_tokenized','txn_sequence','hour_of_day','amount']:
        if c in df.columns: df.drop(columns=c, inplace=True)
    cols = ['class'] + [c for c in df.columns if c != 'class']
    out = df[cols]; out.columns = list(range(len(cols)))   # positional -> matches training
    return out

def output_fn(pred, accept='application/json'):
    proba = model.predict_proba(pred.iloc[:, 1:])[:, 1].tolist()
    payload = {'predictions': [{'score': float(p), 'label': int(p >= 0.5)} for p in proba]}
    if 'application/json' in accept: return json.dumps(payload), 'application/json'
    return ','.join(f'{p:.6f}' for p in proba), 'text/csv'
