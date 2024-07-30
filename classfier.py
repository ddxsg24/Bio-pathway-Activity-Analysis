import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import os

rfc = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=5,
    min_samples_leaf=10,
    min_samples_split=10,
    max_features="sqrt",
    min_impurity_decrease=1e-3
)

file_path = r'D:\code\pathway_research\pathwayscores_final\3365.csv'
file_name = os.path.basename(file_path)
data = pd.read_csv(file_path, header=None)

y = data.iloc[0, 2:].values
y = y.astype(int)

# 删除包含'-INF'或'#NAME?'的行
data = data.replace([-np.inf, np.inf], np.nan)
data = data.replace('#NAME?', np.nan)
data = data.dropna()


X = data.iloc[1:, 2:].T.values
scaler = StandardScaler()
kf = KFold(n_splits=3, shuffle=True, random_state=42)
auc_scores = []  
feature_importance_all_iterations = [] 

for _ in range(500):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    auc_scores.append(auc)
    feature_importance_all_iterations.append(rfc.feature_importances_)


average_auc = np.mean(auc_scores)
average_importance = np.mean(feature_importance_all_iterations, axis=0)

feature_names = data.iloc[1:, 1]
sorted_indices = np.argsort(average_importance)[::-1]
# print(max(sorted_indices))
# print(len(feature_names))
sorted_features = feature_names.iloc[sorted_indices][:15]

print(f"AUC for {file_name} over 500 iterations(DMGI): {average_auc}")
print("Feature Importance Rankings:")
print(sorted_features.values.tolist())
top_15_importance = average_importance[sorted_indices][:15]
print("Top 15 Importance Values:")
print(top_15_importance)


