import os
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import streamlit as st
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load dataset & models
cleaneddf = pd.read_csv(os.path.join(BASE_DIR, "Cleaned.csv"))

#supervised pipelines
NBpipe  = joblib.load(os.path.join(BASE_DIR, "NB.pkl"))
LRpipe  = joblib.load(os.path.join(BASE_DIR, "LR.pkl"))
DTpipe  = joblib.load(os.path.join(BASE_DIR, "DT.pkl"))
LGBMpipe = joblib.load(os.path.join(BASE_DIR, "LGBM.pkl"))
XGBpipe  = joblib.load(os.path.join(BASE_DIR, "XGB.pkl"))
#unsupervised loading
kmeans_artifacts = joblib.load(os.path.join(BASE_DIR, "kmeans_artifacts.pkl"))
kmeans = kmeans_artifacts["kmeans"]
scaler = kmeans_artifacts["scaler"]
numeric_cols = kmeans_artifacts["numeric_cols"]
dummy_columns = kmeans_artifacts["dummy_columns"]
df_cluster_scaled = kmeans_artifacts["train_features"]
labels = kmeans_artifacts["train_labels"]
tsne_train = kmeans_artifacts["tsne_train"]

# Score mapping
score_mapping = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly Agree": 5,
}

# --- Streamlit UI ---
st.title('Mental Health Assessment & Cluster Position')
st.write("Complete the form below. After submitting you'll see your ensemble risk and where you fall among clusters.")

# --- Inputs ---
st.subheader("Personal Information")
age = st.number_input("Age", min_value=18, max_value=100, value=25)
gender = st.selectbox("Gender", ("male", "female", "other", "prefer not to say"))
employment_status = st.selectbox("Employment Status", ("employed", "unemployed", "self-employed", "student"))

st.subheader("Work & Environment")
if employment_status.lower() == "unemployed":
    work_environment = "N/A"
    st.info("Work Environment set to N/A (Unemployed)")
else:
    work_environment = st.selectbox("Work Environment", ("remote", "on-site", "hybrid"))

st.subheader("Mental Health Background")
mental_health_history = st.selectbox("Family Mental Health History", ("yes", "no"))
seeks_treatment = st.selectbox("Currently Seeks Treatment", ("yes", "no"))

st.subheader("Lifestyle Factors")
stress_level = st.number_input("Stress Level (0-10)", min_value=0, max_value=10, value=5)
sleep_hours = st.number_input("Avg Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)
physical_activity_days = st.number_input("Physical Activity Days per Week", min_value=0, max_value=7, value=0)

st.subheader("Assessment Scores (1-5)")
depression_score = st.radio("I feel depressed:", list(score_mapping.keys()))
anxiety_score = st.radio("I feel anxious:", list(score_mapping.keys()))
social_support_score = st.radio("I feel socially supported:", list(score_mapping.keys()))
productivity_score = st.radio("I am productive:", list(score_mapping.keys()))

# --- Submit handler ---
if st.button("Submit for Assessment"):

    input_df = pd.DataFrame({
        'age': [int(age)],
        'gender': [str(gender).lower()],
        'employment_status': [str(employment_status).lower()],
        'work_environment': [str(work_environment).lower()],
        'mental_health_history': [str(mental_health_history).lower()],
        'seeks_treatment': [str(seeks_treatment).lower()],
        'stress_level': [float(stress_level)],
        'sleep_hours': [float(sleep_hours)],
        'physical_activity_days': [int(physical_activity_days)],
        'depression_score': [int(score_mapping[depression_score])],
        'anxiety_score': [int(score_mapping[anxiety_score])],
        'social_support_score': [int(score_mapping[social_support_score])],
        'productivity_score': [int(score_mapping[productivity_score])],
    })

    # --- Ensemble Voting ---
    try:
        nb = NBpipe.predict(input_df)[0]
        lr = LRpipe.predict(input_df)[0]
        dt = DTpipe.predict(input_df)[0]
        lgbm = LGBMpipe.predict(input_df)[0]
        xgb = XGBpipe.predict(input_df)[0]
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        raise

    vote_sum = int(nb + lr + dt + lgbm + xgb)

    st.subheader("Mental Health Risk (Ensemble Voting)")
    if vote_sum < 4:
        st.success("Risk Level: Low")
    elif vote_sum < 8:
        st.warning("Risk Level: Medium")
    else:
        st.error("Risk Level: High")



   
    # use the same dummy columns as in training
    input_df = pd.get_dummies(input_df, drop_first=False, dtype=int)
    input_df = input_df.reindex(columns=dummy_columns, fill_value=0)

    # scale numeric the same way as training
    scaled_input_numeric = scaler.transform(input_df[numeric_cols])

    input_scaled = np.hstack([
        scaled_input_numeric,
        input_df.drop(columns=numeric_cols).values
    ])

    
    # --- PREDICT RISK FOR ALL TRAINING DATA ---
    # Use the ORIGINAL cleaneddf without any preprocessing
    # The models' pipelines will handle all preprocessing
    cleaneddf_copy = cleaneddf.copy()
    train_df_for_pred = cleaneddf_copy.drop(columns=['mental_health_risk'], errors='ignore')
    
    # Predict risk for every row using the raw data (pipelines handle preprocessing)
    nb_pred  = NBpipe.predict(train_df_for_pred)
    lr_pred  = LRpipe.predict(train_df_for_pred)
    dt_pred  = DTpipe.predict(train_df_for_pred)
    lgb_pred = LGBMpipe.predict(train_df_for_pred)
    xgb_pred = XGBpipe.predict(train_df_for_pred)

    cleaneddf_copy["predicted_risk"] = (nb_pred + lr_pred + dt_pred + lgb_pred + xgb_pred)
    cleaneddf_copy["cluster"] = kmeans.labels_

    # --- LABEL CLUSTERS BY RISK LEVEL ---
    cluster_risk_means = cleaneddf_copy.groupby("cluster")["predicted_risk"].mean().sort_values()
    
    # Map clusters to risk levels (lowest mean = Low Risk, highest = High Risk)
    cluster_labels = {}
    risk_level_names = ["Low Risk", "Medium Risk", "High Risk"]
    for idx, cluster_id in enumerate(cluster_risk_means.index):
        cluster_labels[cluster_id] = risk_level_names[idx]
    
    # Get user's cluster and label
    user_cluster = int(kmeans.predict(input_scaled)[0])
    user_cluster_label = cluster_labels[user_cluster]
    low_risk_cluster = int(cluster_risk_means.idxmin())
    
    st.write(f"### Your Cluster: **{user_cluster_label} (Cluster {user_cluster})**")
    st.write(f"### Cluster Risk Levels:")
    for cluster_id, label in cluster_labels.items():
        mean_risk = cluster_risk_means[cluster_id]
        st.write(f"- **Cluster {cluster_id} ({label})**: Average Risk Score = {mean_risk:.2f}")

    # --- t-SNE Visualization ---
    st.write("### t-SNE Visualization")
    train_tsne = tsne_train
    labels = kmeans.labels_

    # Compute centroids in t-SNE space
    centroids_tsne = np.array([train_tsne[labels == i].mean(axis=0) for i in range(3)])

    #Placing User 
    user_2d = centroids_tsne[user_cluster]

    # --- Plot with Risk-Based Labels ---
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
    
    # Map colors by risk level (sorted)
    cluster_colors = {}
    for idx, cluster_id in enumerate(cluster_risk_means.index):
        cluster_colors[cluster_id] = colors[idx]

    # Scatter cluster points
    for i in range(3):
        pts = train_tsne[labels == i]
        if pts.shape[0] > 0:
            label_text = f"{cluster_labels[i]} (Cluster {i})"
            ax.scatter(pts[:, 0], pts[:, 1], c=cluster_colors[i], alpha=0.3, s=45, label=label_text)

            # Draw circles around clusters
            distances = np.linalg.norm(pts - centroids_tsne[i], axis=1)
            radius = np.percentile(distances, 95)
            circle = plt.Circle((centroids_tsne[i, 0], centroids_tsne[i, 1]),
                            radius=radius, edgecolor=cluster_colors[i],
                            facecolor='none', linewidth=2.0, linestyle='--', alpha=0.6, zorder=2)
            ax.add_patch(circle)

    # Plot centroids
    ax.scatter(centroids_tsne[:, 0], centroids_tsne[:, 1],
           marker='X', c='black', s=220, edgecolors='white', linewidth=1.5, label='Centroids')

    # Plot user
    ax.scatter(user_2d[0], user_2d[1],
           c=cluster_colors[user_cluster], s=380, marker='*', edgecolors='black', linewidth=1.3, 
           zorder=5, label=f'You ({user_cluster_label})') ##plot user on T SNE grpah with user pos zorder 5 so star is on top

    ax.set_xlabel('t-SNE Component 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('t-SNE Component 2', fontsize=12, fontweight='bold')
    ax.set_title('Mental Health Risk Clusters', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # --- Recommendations ---
    st.subheader("Recommendations")

    numeric_cols_all = [
        'age', 'stress_level', 'sleep_hours', 'physical_activity_days',
        'depression_score', 'anxiety_score', 'social_support_score',
        'productivity_score'
    ]

    cluster_averages = cleaneddf_copy.groupby("cluster")[numeric_cols_all].mean()
    low_risk_avg = cluster_averages.loc[low_risk_cluster].to_dict()

    tips = []

    user_values = {
        'stress_level': stress_level,
        'sleep_hours': sleep_hours,
        'physical_activity_days': physical_activity_days,
        'depression_score': score_mapping[depression_score],
        'anxiety_score': score_mapping[anxiety_score],
        'social_support_score': score_mapping[social_support_score],
        'productivity_score': score_mapping[productivity_score]
    }

    if user_cluster != low_risk_cluster:
        st.info(f"Recommendations based on differences from the {cluster_labels[low_risk_cluster]} cluster:")

        for feat, user_val in user_values.items():
            if feat not in low_risk_avg:
                continue

            cluster_val = low_risk_avg[feat]
            diff = user_val - cluster_val

            # Negative features (lower = better)
            if feat in ["stress_level", "depression_score", "anxiety_score"]:
                if diff > 0.5:
                    tips.append(
                        f"Improve {feat.replace('_',' ').title()}: "
                        f"Your score ({user_val}) is higher than low-risk avg ({cluster_val:.1f})."
                    )

            # Positive features (higher = better)
            else:
                if diff < -0.5:
                    tips.append(
                        f"**Increase {feat.replace('_',' ').title()}:** "
                        f"You ({user_val}) are below low-risk avg ({cluster_val:.1f})."
                    )

        if seeks_treatment.lower() == 'no' and vote_sum >= 4:
            tips.append(" Your assessed risk is elevated. Consider a professional Evaluation")

    else:
        st.success(f"You already belong to the {cluster_labels[low_risk_cluster]} cluster!")
        tips.append("**Continue your positive habits!**")

    for t in tips:
        st.markdown(f"- {t}")