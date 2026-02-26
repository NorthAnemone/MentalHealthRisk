import os
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
import lightgbm as lgb
import sklearn
from sklearn.manifold import TSNE

print("Using scikit-learn version:", sklearn.__version__)

#dataset loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(BASE_DIR, "Cleaned.csv")
Dataset = pd.read_csv(csv_path)


#turns all environments of unemployed people to N/A
Dataset.loc[Dataset["employment_status"] == "Unemployed", "work_environment"] = "N/A"

#SUPERVISED LEARNING
#mapping the label
mapping = {"low": 0, "medium": 1, "high": 2}
Data = Dataset.drop("mental_health_risk", axis=1)
Labels = Dataset["mental_health_risk"].map(mapping)

#train/test
Data_train, Data_test, Label_train, Label_test = train_test_split(
    Data, Labels, test_size=0.2, random_state=42
)

#preprocessing
categorical_cols = Data.select_dtypes(include=["object"]).columns
numeric_cols = Data.select_dtypes(exclude=["object"]).columns

preprocess = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("numeric", StandardScaler(), numeric_cols),
    ]
)

#pipelines
NBpipe = Pipeline([
    ("preprocess", preprocess),
    ("nb", GaussianNB())
])

LRpipe = Pipeline([
    ("preprocess", preprocess),
    ("lr", LogisticRegression(max_iter=200))
])

DTpipe = Pipeline([
    ("preprocess", preprocess),
    ("dt", DecisionTreeClassifier(max_depth=5, random_state=42))
])

LGBMpipe = Pipeline([
    ("preprocess", preprocess),
    ("lgbm", lgb.LGBMClassifier(n_estimators=100, learning_rate=0.09,random_state=42))
])


XGBpipe = Pipeline([
    ("preprocess", preprocess),
    ("xgb", xgb.XGBClassifier(n_estimators=100, learning_rate=0.09,random_state=42))
])

NBpipe.fit(Data_train, Label_train)
LRpipe.fit(Data_train, Label_train)
DTpipe.fit(Data_train, Label_train)
LGBMpipe.fit(Data_train, Label_train)
XGBpipe.fit(Data_train, Label_train)

#check for accuracy
#Prediction and accuracy showing, confusion matrix is:
#Low: Low,Medium,High
#Medium: Low,Medium,High
#High: Low,Medium,High
NB_predict = NBpipe.predict(Data_test)
print("Accuracy: ", accuracy_score(Label_test, NB_predict))
print("\nConfusion Matrix Naive Bayes: \n",confusion_matrix(Label_test, NB_predict))
print("\nClassification Report: \n", classification_report(Label_test, NB_predict))

LR_predict = LRpipe.predict(Data_test)
print("Accuracy: ", accuracy_score(Label_test, LR_predict))
print("\nConfusion Matrix Logistical Regression: \n",confusion_matrix(Label_test, LR_predict))
print("\nClassification Report: \n", classification_report(Label_test, LR_predict))

DT_predict = DTpipe.predict(Data_test)
print("Accuracy: ", accuracy_score(Label_test, DT_predict))
print("\nConfusion Matrix Decision Trees: \n",confusion_matrix(Label_test, DT_predict))
print("\nClassification Report: \n", classification_report(Label_test, DT_predict))

LGBM_predict = LGBMpipe.predict(Data_test)
print("Accuracy: ", accuracy_score(Label_test, LGBM_predict))
print("\nConfusion Matrix Decision Trees: \n",confusion_matrix(Label_test, LGBM_predict))
print("\nClassification Report: \n", classification_report(Label_test, LGBM_predict))

XGB_predict = XGBpipe.predict(Data_test)
print("Accuracy: ", accuracy_score(Label_test, XGB_predict))
print("\nConfusion Matrix Decision Trees: \n",confusion_matrix(Label_test, XGB_predict))
print("\nClassification Report: \n", classification_report(Label_test, XGB_predict))
# Save model pickles
joblib.dump(NBpipe,  os.path.join(BASE_DIR, "NB.pkl"))
joblib.dump(LRpipe,  os.path.join(BASE_DIR, "LR.pkl"))
joblib.dump(DTpipe,  os.path.join(BASE_DIR, "DT.pkl"))
joblib.dump(LGBMpipe, os.path.join(BASE_DIR, "LGBM.pkl"))
joblib.dump(XGBpipe,  os.path.join(BASE_DIR, "XGB.pkl"))

#UNSUPERVISED LEARNING
print("Preparing KMeans clustering...")

#Removes label
cluster_df = Dataset.drop(columns=["mental_health_risk"])
cluster_df.columns = [c.strip().lower() for c in cluster_df.columns]

#One-hot encode categorial
cluster_df_dummies = pd.get_dummies(cluster_df, drop_first=False, dtype=int)

#Numeric features used for StandardScaler
kmeans_numeric = [
    "age", "stress_level", "sleep_hours", "physical_activity_days",
    "depression_score", "anxiety_score", "social_support_score",
    "productivity_score"
]

# Fit scaler on numeric columns
scaler = StandardScaler()
scaled_numeric = scaler.fit_transform(cluster_df_dummies[kmeans_numeric])

# Combine numeric (scaled) + all dummy categorical
cluster_features = np.hstack([
    scaled_numeric,
    cluster_df_dummies.drop(columns=kmeans_numeric).values
])

#Fit KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
kmeans.fit(cluster_features)


#Save ALL needed artifacts in one file
artifacts = {
    "kmeans": kmeans,
    "scaler": scaler,
    "numeric_cols": kmeans_numeric,
    "dummy_columns": cluster_df_dummies.columns.tolist(),
    "train_features": cluster_features,
    "train_labels": kmeans.labels_,
}


#adds the TSNE to the kmeans pkl, makes them one pkl file

print("Getting T-SNE Features...") 

tsne = TSNE(n_components=2, perplexity=min(30, max(5, len(cluster_features)// 3)), init="pca", random_state=42)
tsne_train = tsne.fit_transform(cluster_features)

artifacts["tsne_train"] = tsne_train
joblib.dump(artifacts, os.path.join(BASE_DIR, "kmeans_artifacts.pkl"))