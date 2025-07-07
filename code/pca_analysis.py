# pca_analysis.py
# PCA and Clustering Visualization

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Load data
input_folder = './datasets/'
labevents = pd.read_csv(input_folder + 'labevents.csv')
admissions = pd.read_csv(input_folder + 'mimc_organize.csv')

# Preprocessing
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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

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
                    break
            else:
                truncation_counter = 0
    shapley_vals /= num_iterations
    return shapley_vals

# Refinement and filtering
version_storage = {}
version_flags = {
    'Baseline': {'if': False, 'odds': False},
    'IF only': {'if': True, 'odds': False},
    'Odds only': {'if': False, 'odds': True},
    'IF + Odds': {'if': True, 'odds': True}
}

X_train = pd.DataFrame(X_train)
y_train = pd.Series(y_train)

for version, flags in version_flags.items():
    X_curr, y_curr = X_train.copy(), y_train.copy()
    X_bin = (X_curr > 0).astype(int)
    or_dict = compute_odds_ratio_binary(X_bin, y_curr)

    if flags['odds'] == True:
        filtered_idx = []
        for i in range(len(X_curr)):
            row = X_bin.iloc[i]
            if y_curr.iloc[i] == 0:
                if any(or_dict[f] > 1.2 and row[f] == 1 for f in X_bin.columns):
                    filtered_idx.append(i)
            else:
                if any(or_dict[f] < -1.2 and row[f] == 1 for f in X_bin.columns):
                    filtered_idx.append(i)
        X_curr = X_curr.iloc[filtered_idx].reset_index(drop=True)
        y_curr = y_curr.iloc[filtered_idx].reset_index(drop=True)

    if flags['if']:
        weights = {f: 1 for f in X_curr.columns}
        mask = get_if_mask(X_curr, weights)
        X_curr, y_curr = X_curr[mask], y_curr[mask]

    shap_vals = calculate_shapley(X_curr.values, y_curr.values, X_test, y_test)
    threshold = np.percentile(shap_vals, 40)
    X_filt = X_curr[shap_vals >= threshold]
    version_storage[version] = X_filt

# Visualization
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_kmeans_clusters(X_data, version_name, sample_size=5000):
    if not isinstance(X_data, pd.DataFrame):
        X_data = pd.DataFrame(X_data)
    if len(X_data) > sample_size:
        X_data = X_data.sample(n=sample_size, random_state=42)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_data)
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X_pca)
    df_plot = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
    df_plot["Cluster"] = clusters
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df_plot, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", alpha=0.7)
    plt.title(f"{version_name} - PCA + KMeans (k=2)")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.legend(title="Cluster ID")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Run visualizations
for version in version_storage:
    visualize_kmeans_clusters(version_storage[version], version)
