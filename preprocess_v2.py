import argparse, os, glob
import pandas as pd, numpy as np
from sklearn.preprocessing import RobustScaler

def preprocess(input_dir, output_dir):
    files = glob.glob(os.path.join(input_dir, '*.parquet'))
    assert files, f"No parquet in {input_dir}: {os.listdir(input_dir)}"
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df.columns = [str(c).lower() for c in df.columns]
    print("Loaded", df.shape, list(df.columns))
    missing = [c for c in ['class','amount','hour_of_day'] if c not in df.columns]
    assert not missing, f"Missing {missing}; found {list(df.columns)}"

    if 'txn_sequence' in df.columns:
        df = df.sort_values('txn_sequence').reset_index(drop=True)

    df['hour_sin'] = np.sin(2*np.pi*df['hour_of_day']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour_of_day']/24)

    n = len(df); tr = int(n*0.70); va = int(n*0.15)
    train_df = df.iloc[:tr].copy(); val_df = df.iloc[tr:tr+va].copy(); test_df = df.iloc[tr+va:].copy()

    sc = RobustScaler()
    train_df['amount_scaled'] = sc.fit_transform(train_df[['amount']])
    val_df['amount_scaled']   = sc.transform(val_df[['amount']])
    test_df['amount_scaled']  = sc.transform(test_df[['amount']])

    drop = [c for c in ['time_tokenized','txn_sequence','hour_of_day','amount'] if c in df.columns]
    for p in (train_df, val_df, test_df): p.drop(columns=drop, inplace=True)

    cols = ['class'] + [c for c in train_df.columns if c != 'class']
    for name, part in [('train',train_df),('validation',val_df),('test',test_df)]:
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)
        part[cols].to_csv(os.path.join(output_dir, name, name + '.csv'), index=False, header=False)
        print(name, part.shape, "fraud%", round(part['class'].mean()*100, 4))
    print("Features:", cols[1:])

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input-data', type=str, default='/opt/ml/processing/input')
    p.add_argument('--output-data', type=str, default='/opt/ml/processing/output')
    a = p.parse_args(); preprocess(a.input_data, a.output_data)
