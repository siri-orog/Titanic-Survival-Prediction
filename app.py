# final_titanic_app.py
# Titanic Survival Prediction - Final Project App
# Author: (Your Name)
# Run with: streamlit run final_titanic_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# ---------------------------- Streamlit Setup ----------------------------
st.set_page_config(page_title="Titanic Survival Prediction (Final)", layout="wide")
st.title("🚢 Titanic Survival Prediction — Final Project")

st.markdown("""
**What this app does:**  
Upload any Titanic-style dataset (CSV/XLSX).  
It will preprocess, run EDA, train multiple models, evaluate them (% metrics), pick the best one,  
allow predictions, and let you download the trained model artifact.
""")

# ---------------------------- Utility Functions ----------------------------
def find_target_column(df):
    for c in df.columns:
        if c.lower() == "survived":
            return c
    for c in df.columns:
        vals = df[c].dropna().unique()
        if set(vals).issubset({0, 1}):
            return c
    return None

def simple_preprocess(df_in):
    df = df_in.copy()
    drop_candidates = ["PassengerId", "Name", "Ticket", "Cabin"]
    for c in drop_candidates:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # Fill missing
    for col in df.columns:
        if df[col].dtype == "object":
            if df[col].isnull().any():
                try:
                    df[col].fillna(df[col].mode()[0], inplace=True)
                except:
                    df[col].fillna("missing", inplace=True)
        else:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)

    # Encode categoricals
    encoders = {}
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str)
        uniques = df[col].unique().tolist()
        mapping = {v: i for i, v in enumerate(uniques)}
        df[col] = df[col].map(mapping)
        encoders[col] = mapping

    return df, encoders

def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }

# ---------------------------- File Upload ----------------------------
uploaded = st.file_uploader("Upload Titanic dataset (.csv or .xlsx)", type=["csv", "xlsx"])

if not uploaded:
    st.info("Upload dataset to start. Use Kaggle Titanic train.csv or any similar file.")
    st.stop()

try:
    if uploaded.name.endswith(".csv"):
        df_raw = pd.read_csv(uploaded)
    else:
        df_raw = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

st.subheader("🗂 Dataset Preview")
st.dataframe(df_raw.head())
st.info(f"Dataset contains **{df_raw.shape[0]} rows** and **{df_raw.shape[1]} columns**.")

target_col = find_target_column(df_raw)
if not target_col:
    st.error("Could not detect the target column (Survived).")
    st.stop()
st.success(f"Detected target column: **{target_col}**")

# ---------------------------- Preprocessing ----------------------------
st.header("🧹 Preprocessing")
st.write("Missing values (before):")
st.dataframe(df_raw.isnull().sum())

df_proc, encoders = simple_preprocess(df_raw)

st.write("Missing values (after):")
st.dataframe(df_proc.isnull().sum())

st.write("Missing values visualization:")
fig = plt.figure(figsize=(5, 2.5))
msno.bar(df_raw, fontsize=8)
st.pyplot(fig)
plt.clf()

if target_col not in df_proc.columns:
    st.error("Target column removed during preprocessing.")
    st.stop()

X = df_proc.drop(columns=[target_col])
y = df_proc[target_col]

st.write("Features used for modeling:")
st.write(list(X.columns))

# ---------------------------- EDA ----------------------------
# ---------------------------- EDA ----------------------------
st.header("🔎 Compact EDA")

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.subheader("Numerical Summary")
    st.dataframe(X.describe().T)

with col2:
    st.subheader("Target Distribution")
    fig, ax = plt.subplots(figsize=(3, 2.5))
    sns.countplot(x=y, palette="Set2", ax=ax)
    ax.set_xlabel(target_col)
    st.pyplot(fig)
    plt.clf()

with col3:
    st.subheader("Correlation Heatmap")
    num = df_proc.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    sns.heatmap(num.corr(), annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    plt.clf()

# ---------------------------- Pattern Detection ----------------------------
st.subheader("🧠 Automatic Pattern Detection (Text-Based Insights)")

insights = []

# 1️⃣ Correlation-based patterns
corr = num.corr()[target_col].sort_values(ascending=False)
top_pos = corr[corr > 0.1].drop(target_col, errors="ignore")
top_neg = corr[corr < -0.1]

if not top_pos.empty:
    st.markdown("### 🔹 Strong Positive Correlations")
    for feat, val in top_pos.items():
        st.markdown(f"- **{feat}** → +{val:.2f} correlation with survival")
else:
    st.markdown("No strong positive correlations found.")

if not top_neg.empty:
    st.markdown("### 🔸 Strong Negative Correlations")
    for feat, val in top_neg.items():
        st.markdown(f"- **{feat}** → {val:.2f} correlation with survival")
else:
    st.markdown("No strong negative correlations found.")

# 2️⃣ Overall survival rate
df_temp = df_proc.copy()
survival_rate = df_temp[target_col].mean() * 100
st.markdown(f"### 🧮 Overall Survival Rate: **{survival_rate:.2f}%**")

# 3️⃣ Categorical patterns
st.markdown("### 📌 Survival Rates by Categorical Features")

for col in df_raw.select_dtypes(include=['object']).columns:
    if col in df_temp.columns:
        try:
            rate = df_temp.groupby(col)[target_col].mean().sort_values(ascending=False) * 100
            st.markdown(f"**{col}:**")
            for cat, r in rate.items():
                st.markdown(f"- {cat}: {r:.2f}%")
        except:
            pass

# 4️⃣ Numeric feature patterns
st.markdown("### 📊 Numeric Feature Effects")

for col in df_temp.select_dtypes(include=[np.number]).columns:
    if col != target_col:
        try:
            high = df_temp[df_temp[col] >= df_temp[col].median()][target_col].mean() * 100
            low = df_temp[df_temp[col] < df_temp[col].median()][target_col].mean() * 100
            st.markdown(
                f"- **{col}**: Higher values → {high:.2f}%, Lower values → {low:.2f}%"
            )
        except:
            pass

# ---------------------------- Model Training & Evaluation ----------------------------
st.header("🤖 Train & Compare Multiple Models")
metric = st.selectbox("Pick primary metric to choose best model", options=["accuracy","f1","percision","recall","cv"],index=0)

test_size = st.slider("Test set size (fraction)", 10, 40, 20) / 100.0
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=test_size, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = build_models()

# --- NEW MODEL TRAINING USING CROSS-VALIDATION ---
results = []
st.write("Running 10-fold Cross-Validation for each model...")

for name, model in models.items():
    # 10-fold cross validation
    scores = cross_val_score(model, np.vstack((X_train, X_test)), np.concatenate((y_train, y_test)), cv=10)
    mean_acc = scores.mean() * 100
    std_acc = scores.std() * 100

    # Train-test evaluation
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    results.append({
        "model": name,
        "accuracy (%)": round(acc * 100, 2),
        "cv mean (%)": round(mean_acc, 2),
        "cv std (%)": round(std_acc, 2),
        "f1 (%)": round(f1 * 100, 2),
        "precision (%)": round(prec * 100, 2),
        "recall (%)": round(rec * 100, 2)
    })

res_df = pd.DataFrame(results).sort_values(by="cv mean (%)", ascending=False)

st.subheader("📊 Model Performance (in %)")
st.dataframe(res_df.reset_index(drop=True))

# Bar chart for visual comparison (Compact)
fig, ax = plt.subplots(figsize=(4, 2))  # 🔹 Smaller and more window-friendly
sns.barplot(x="model", y="accuracy (%)", data=res_df, ax=ax, palette="Set2", edgecolor="black")
ax.set_ylim(70, 100)
ax.set_title("Model Accuracy Comparison", fontsize=10)
ax.set_xlabel("Model", fontsize=9)
ax.set_ylabel("Accuracy (%)", fontsize=9)
plt.xticks(rotation=20, fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout(pad=0.5)
st.pyplot(fig, use_container_width=False)
plt.clf()

best_model_name = res_df.iloc[0]["model"]
best_model = models[best_model_name]
st.success(f"🏆 Best model: {best_model_name}")

# Retrain best model on full data
best_model.fit(np.vstack((X_train, X_test)), np.concatenate((y_train, y_test)))

# ---------------------------- Evaluation Visuals ----------------------------
st.header("📈 Best Model Evaluation")

y_pred_test = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_test)
fig, ax = plt.subplots(figsize=(2.8, 2))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, annot_kws={"size": 8})
ax.set_xlabel("Predicted", fontsize=9)
ax.set_ylabel("Actual",fontsize=9 )
ax.tick_params(labelsize=8)
plt.tight_layout(pad=0.5)
st.pyplot(fig, use_container_width=False)
plt.clf()

# Classification report (%)
st.subheader("Detailed Classification Report (%)")
report = classification_report(y_test, y_pred_test, output_dict=True)
report_df = pd.DataFrame(report).T
report_df[["precision", "recall", "f1-score"]] = report_df[
    ["precision", "recall", "f1-score"]
].applymap(lambda x: round(x * 100, 2) if isinstance(x, (float, np.floating)) else x)
st.dataframe(report_df)

# ROC Curve (Compact)
if hasattr(best_model, "predict_proba"):
    y_prob = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(3, 2))  # 🔹 Adjusted size for better fit
    ax.plot(fpr, tpr, color="darkorange", lw=1.5, label=f"AUC = {roc_auc * 100:.2f}%")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)
    ax.tick_params(labelsize=8)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=False)
    plt.clf()

# Feature Importance (Compact)
if "forest" in best_model_name.lower() or "tree" in best_model_name.lower():
    importances = best_model.feature_importances_
    feat_names = X.columns
    fi_df = (
        pd.DataFrame({"feature": feat_names, "importance": importances})
        .sort_values("importance", ascending=False)
    )

    st.subheader("🌳 Feature Importance (Best Model)")

    fig, ax = plt.subplots(figsize=(3, 2))  # 🔹 Compact but readable
    sns.barplot(
        x="importance",
        y="feature",
        data=fi_df,
        ax=ax,
        palette="crest",
        edgecolor="black"
    )
    ax.set_xlabel("Importance", fontsize=9)
    ax.set_ylabel("Feature", fontsize=9)
    ax.tick_params(labelsize=8)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=False)
    plt.clf()


# ---------------------------- Save Model ----------------------------
st.header("💾 Save & Download Best Model")
artifact = {
    "model": best_model,
    "scaler": scaler,
    "encoders": encoders,
    "feature_names": X.columns.tolist()
}
pickle_bytes = pickle.dumps(artifact)
st.download_button("⬇️ Download trained model artifact (.pkl)",
                   data=pickle_bytes, file_name="titanic_best_artifact.pkl")

# ---------------------------- Prediction ----------------------------
st.header("🔮 Predict for a New Passenger")
st.markdown("### Enter passenger details (inputs are limited to realistic values)")

pclass = st.selectbox("Pclass (Passenger Class)", options=[1, 2, 3], index=0)
sex_label = st.selectbox("Sex", options=["male", "female"])
sex = 1 if sex_label == "male" else 0
age = st.slider("Age", min_value=0, max_value=80, value=25)
sibsp = st.number_input("SibSp (Siblings/Spouses Aboard)", min_value=0, max_value=8, value=0, step=1)
parch = st.number_input("Parch (Parents/Children Aboard)", min_value=0, max_value=6, value=0, step=1)
fare = st.number_input("Fare (Ticket Price in $)", min_value=0, max_value=600, value=30, step=1)
embarked_label = st.selectbox("Embarked", options=["S", "C", "Q"])
embarked_map = {"S": 0, "C": 1, "Q": 2}
embarked = embarked_map[embarked_label]

input_vals = {
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked
}

if st.button("Predict"):
    try:
        x_in = [input_vals[c] for c in X.columns]
        x_arr = np.array(x_in).reshape(1, -1)
        x_scaled = scaler.transform(x_arr)

        model = artifact["model"]
        pred = model.predict(x_scaled)[0]
        prob = model.predict_proba(x_scaled)[0][1] * 100 if hasattr(model, "predict_proba") else 0

        if pred == 1:
            st.success(f"✅ Prediction: Survived — Probability {prob:.2f}%")
        else:
            st.error(f"❌ Prediction: Did Not Survive — Probability {prob:.2f}%")

    except Exception as e:
        st.error(f"Failed to predict. Check inputs. Error: {e}")

# ---------------------------- Source Code Viewer ----------------------------
with st.expander("🧾 Click to View Source Code"):
    with open(__file__, encoding="utf-8") as f:
        code = f.read()
    st.code(code, language="python")


