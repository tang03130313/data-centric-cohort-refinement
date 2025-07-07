# main_pipeline.py
# Cell 1: Data Preprocessing (Full Lab + AKI Labeling + ItemID Filtering)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

input_folder = './datasets/'
labevents = pd.read_csv(input_folder + 'labevents.csv')
admissions = pd.read_csv(input_folder + 'mimc_organize.csv')

labevents['charttime'] = pd.to_datetime(labevents['charttime'], errors='coerce')
admissions['admittime'] = pd.to_datetime(admissions['admittime'], errors='coerce')

creatinine_itemid = 50912
creatinine_data = labevents[labevents['itemid'] == creatinine_itemid].dropna(subset=['valuenum'])
creatinine_data['baseline'] = creatinine_data.groupby('subject_id')['valuenum'].transform('min')
creatinine_data['relative_AKI'] = (creatinine_data['valuenum'] / creatinine_data['baseline'] >= 1.5).astype(int)
creatinine_data['change_2d'] = creatinine_data.groupby('subject_id')['valuenum'].diff(1)
creatinine_data['absolute_AKI'] = (creatinine_data['change_2d'] >= 26.5).astype(int)
creatinine_data['AKI'] = creatinine_data[['relative_AKI', 'absolute_AKI']].max(axis=1)

aki_labels = creatinine_data.groupby('subject_id')['AKI'].max().reset_index()

lab_features = labevents.dropna(subset=['valuenum'])
aki_ids = aki_labels[aki_labels['AKI'] == 1]['subject_id']
aki_labs = lab_features[lab_features['subject_id'].isin(aki_ids)]
itemid_counts = aki_labs.groupby('itemid')['subject_id'].nunique()
itemid_threshold = len(aki_ids) * 0.5
valid_itemids = itemid_counts[itemid_counts >= itemid_threshold].index
lab_features = lab_features[lab_features['itemid'].isin(valid_itemids)]

lab_agg = lab_features.groupby(['subject_id', 'itemid'])['valuenum'].mean().unstack(fill_value=0).reset_index()
data = pd.merge(lab_agg, aki_labels, on='subject_id', how='inner')
X = data.drop(columns=['subject_id', 'AKI'])
y = data['AKI']

X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
X = np.log1p(X)
X = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan).fillna(0)
X_scaled = MinMaxScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train = pd.DataFrame(X_train).reset_index(drop=True)
X_test = pd.DataFrame(X_test).reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Cell 2: Helper Functions
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

def compute_odds_ratio_binary(X_bin, y, eps=1e-6):
    odds_ratios = {}
    for col in X_bin.columns:
        a = ((X_bin[col] == 1) & (y == 1)).sum()
        b = ((X_bin[col] == 1) & (y == 0)).sum()
        c = ((X_bin[col] == 0) & (y == 1)).sum()
        d = ((X_bin[col] == 0) & (y == 0)).sum()
        OR = ((a + eps) * (d + eps)) / ((b + eps) * (c + eps))
        odds_ratios[col] = np.log(OR)
    return odds_ratios

def get_if_mask(X, weights):
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    iso.fit(X)
    scores = iso.decision_function(X)
    total_score = np.zeros_like(scores)
    for i, f in enumerate(X.columns):
        total_score += scores * weights[f]
    threshold = np.percentile(total_score, 1)
    return total_score > threshold

def calculate_shapley(X_train, y_train, X_test, y_test, num_iterations=10, truncation_tolerance=0.001, early_stop_threshold=5, verbose=True):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_proba = lr.predict_proba(X_test)[:, 1]
    full_auc = roc_auc_score(y_test, y_proba)
    if verbose:
        print(f"[Shapley] Full AUC: {full_auc:.4f}")
    shapley_vals = np.zeros(len(X_train))
    for iter_idx in tqdm(range(num_iterations), desc="Monte Carlo Iterations"):
        perm = np.random.permutation(len(X_train))
        old_auc = full_auc
        truncation_counter = 0
        for idx in perm:
            mask = np.ones(len(X_train), dtype=bool)
            mask[idx] = False
            X_temp, y_temp = X_train[mask], y_train[mask]
            lr.fit(X_temp, y_temp)
            new_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
            shapley_vals[idx] += old_auc - new_auc
            old_auc = new_auc
            if abs(new_auc - full_auc) <= truncation_tolerance * full_auc:
                truncation_counter += 1
                if truncation_counter >= early_stop_threshold:
                    if verbose:
                        print(f"[Shapley] Early stopped after {truncation_counter} samples within tolerance.")
                    break
            else:
                truncation_counter = 0
    shapley_vals /= num_iterations
    return shapley_vals

def build_mlp(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model

def evaluate_model(model, X_test, y_test, name, version):
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob > 0.5).astype(int)
    return {
        'Version': version,
        'Model': name,
        'AUC': roc_auc_score(y_test, y_prob),
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'FPR': roc_curve(y_test, y_prob)[0],
        'TPR': roc_curve(y_test, y_prob)[1],
        'Confusion': confusion_matrix(y_test, y_pred)
    }

# Cell 3: Run All Versions including updated Log(OR) Only
from sklearn.model_selection import StratifiedKFold

versions = {
    'Baseline': {'if': False, 'odds': None},
    'IF only': {'if': True, 'odds': None},
    'IF + Odds': {'if': True, 'odds': 'split'},
    'OR Only': {'if': False, 'odds': 'split'}
}

results = []
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold_idx, (train_idx, test_idx) in enumerate(folds.split(X, y)):
    print(f"\n==== Fold {fold_idx + 1} ====")
    X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)

    for version, flags in versions.items():
        print(f"\n-- {version} --")
        X_curr, y_curr = X_train.copy(), y_train.copy()
        X_bin = (X_curr > 0).astype(int)
        or_dict = compute_odds_ratio_binary(X_bin, y_curr)

        if flags['odds'] == 'split':
            neg_mask = X_bin[y_curr == 0].apply(
                lambda row: any(or_dict[col] <= 3 and row[col] == 1 for col in X_bin.columns), axis=1)
            pos_mask = X_bin[y_curr == 1].apply(
                lambda row: any(or_dict[col] >= -3 and row[col] == 1 for col in X_bin.columns), axis=1)
            neg_idx = y_curr[y_curr == 0].index[neg_mask]
            pos_idx = y_curr[y_curr == 1].index[pos_mask]
            keep_idx = neg_idx.union(pos_idx)
            X_curr, y_curr = X_curr.loc[keep_idx], y_curr.loc[keep_idx]
            weights = {f: 1 for f in X_curr.columns}
        else:
            weights = {f: 1 for f in X_bin.columns}

        if flags['if']:
            mask = get_if_mask(X_curr, weights)
            X_curr, y_curr = X_curr[mask], y_curr[mask]

        # SHAP FILTER
        shap_vals = calculate_shapley(X_curr.values, y_curr.values, X_test.values, y_test.values)
        shap_threshold = np.percentile(shap_vals, 40)
        shap_mask = shap_vals > shap_threshold
        X_filt, y_filt = X_curr[shap_mask], y_curr[shap_mask]

        for name, model in {
            'Logistic': LogisticRegression(max_iter=1000),
            'RF': RandomForestClassifier(n_estimators=100),
            'XGB': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            'MLP': build_mlp(X_filt.shape[1])
        }.items():
            if name == 'MLP':
                model.fit(X_filt, y_filt, epochs=50, verbose=0)
            else:
                model.fit(X_filt, y_filt)
            res = evaluate_model(model, X_test, y_test, name, version + ' - SHAP Filter')
            res['Fold'] = fold_idx + 1
            results.append(res)

        # SHAP WEIGHT
        shap_norm = MinMaxScaler().fit_transform(shap_vals.reshape(-1, 1)).ravel()
        sample_weight = np.ones(len(y_curr))
        sample_weight[y_curr == 0] = 1 - shap_norm[y_curr == 0]

        for name, model in {
            'Logistic': LogisticRegression(max_iter=1000),
            'RF': RandomForestClassifier(n_estimators=100),
            'XGB': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            'MLP': build_mlp(X_curr.shape[1])
        }.items():
            if name == 'MLP':
                model.fit(X_curr, y_curr, sample_weight=sample_weight, epochs=50, verbose=0)
            else:
                model.fit(X_curr, y_curr, sample_weight=sample_weight)
            res = evaluate_model(model, X_test, y_test, name, version + ' - SHAP Weight')
            res['Fold'] = fold_idx + 1
            results.append(res)


# Cell 4: Visualize Results
import matplotlib.pyplot as plt
import seaborn as sns

results_df = pd.read_pickle("_results/version11_20250706_v5A_good.pkl")
summary_df = results_df.groupby(['Version', 'Model'])[['AUC', 'Accuracy', 'F1', 'Precision', 'Recall']].agg(['mean', 'std'])
pd.set_option('display.float_format', '{:.4f}'.format)
print(summary_df)

# Save results
results_df.to_csv("_results/version11_20250706_v5A_good.csv", index=False)
summary_df.to_latex("_results/version11_20250706_v5A_good.tex", float_format="%.4f")

# ROC Curves
for model in results_df['Model'].unique():
    plt.figure(figsize=(6, 4))
    for version in results_df['Version'].unique():
        subset = results_df[(results_df['Model'] == model) & (results_df['Version'] == version)]
        for _, row in subset.iterrows():
            plt.plot(row['FPR'], row['TPR'], alpha=0.3)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"ROC Curves - {model} across 5 folds")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.legend(results_df['Version'].unique(), title='Version', loc='lower right')
    plt.tight_layout()
    plt.show()
