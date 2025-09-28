import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import time

# Import models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def load_data(file_path):
    """Loads the dataset from a CSV file."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please make sure it's in the same directory.")
        return None

def preprocess_data(df, addiction_threshold):
    """Preprocesses the data for model training."""
    # Create binary target variable based on the addiction threshold
    df['Addicted'] = (df['Addiction_Level'] >= addiction_threshold).astype(int)

    # Drop columns that are identifiers or have been used to create the target
    df = df.drop(['ID', 'Name', 'Location', 'Addiction_Level'], axis=1)

    # Define features (X) and target (y)
    X = df.drop('Addicted', axis=1)
    y = df['Addicted']

    # Identify categorical and numerical features for transformations
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

    # Create a preprocessor object using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test, preprocessor

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Algorithm Comparison", layout="wide")

    st.title("📱 Supervised Learning for Teen Smartphone Addiction")
    st.write("""
    This application compares various supervised machine learning algorithms for predicting smartphone addiction among teenagers.
    The models are trained on the 'Teen Smartphone Usage and Addiction Impact Dataset'. You can adjust the addiction threshold and see how it affects model performance.
    """)

    # --- Sidebar for User Inputs ---
    st.sidebar.header("⚙️ Configuration")
    addiction_threshold = st.sidebar.slider(
        "Addiction Level Threshold", 
        min_value=1.0, 
        max_value=10.0, 
        value=7.5, 
        step=0.1,
        help="A user with an 'Addiction_Level' at or above this value will be classified as 'Addicted'."
    )
    
    # --- Load and Process Data ---
    df_original = load_data('teen_phone_addiction_dataset (1).csv')
    if df_original is None:
        return
        
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df_original.copy(), addiction_threshold)

    # --- Model Definitions ---
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'),
        "k-Nearest Neighbors (kNN)": KNeighborsClassifier(),
        "Support Vector Machine (SVM)": SVC(probability=True, random_state=42),
        "Neural Network": MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(50,)),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    # --- Train and Evaluate ---
    results_list = []
    with st.spinner("Training models and calculating metrics..."):
        for name, model in models.items():
            start_time = time.time()
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
            pipeline.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

            results_list.append({
                "Algorithm": name,
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1-Score": f1_score(y_test, y_pred),
                "AUC-ROC": roc_auc_score(y_test, y_pred_proba),
                "Training Time (s)": training_time
            })
    
    metrics_df = pd.DataFrame(results_list).set_index("Algorithm")

    # --- Prepare Data for Display ---
    st.info("""
    **Note on K-Means:** You listed K-Means, which is an **unsupervised clustering** algorithm. Its purpose is to group data without predefined labels. 
    Since this task is about predicting a specific label ('Addicted' vs. 'Not Addicted'), it is a **supervised classification** problem. Therefore, K-Means is not applicable here and has been excluded from the comparison.
    """)

    qualitative_data = {
        'Algorithm': list(models.keys()),
        'Bias–Variance': ['Low Bias, High Variance', 'High Bias, Low Variance', 'Low Bias, High Variance', 'Tunable (Low Bias, High Var w/ complex kernels)', 'Low Bias, High Variance', 'Low Bias, Medium Variance'],
        'Data Size Sensitivity': ['Moderate', 'Performs well on varied sizes', 'Sensitive to large datasets (slow)', 'Can be slow on very large datasets', 'Requires large datasets for best results', 'Scales well'],
        'Training Time': ['Fast', 'Fast', 'None (Lazy Learner)', 'Slow with non-linear kernels', 'Very Slow', 'Moderate to Slow'],
        'Prediction Speed': ['Fast', 'Fast', 'Slow', 'Fast', 'Fast', 'Moderate'],
        'Memory Usage': ['Low', 'Low', 'High (stores all data)', 'Moderate', 'High', 'High (stores many trees)'],
        'Hyperparameter Sensitivity': ['Sensitive (e.g., max_depth)', 'Less Sensitive (e.g., C)', 'Sensitive (k)', 'Very Sensitive (C, kernel, gamma)', 'Very Sensitive (architecture, learning rate)', 'Less Sensitive than one tree'],
        'Robustness': ['Low (sensitive to data changes)', 'Robust with regularization', 'Low (sensitive to outliers)', 'High (maximizes margin)', 'Can be robust with good design', 'Very High (ensemble method)'],
        'Best-Suited Metrics': ['F1-Score, Interpretability', 'AUC-ROC, good baseline', 'Accuracy (if balanced)', 'AUC-ROC, powerful in high dimensions', 'AUC-ROC, Recall (for complex patterns)', 'AUC-ROC, F1-Score (strong all-rounder)'],
        'Remarks': ['Highly interpretable and a good starting point. Prone to overfitting.', 'Excellent baseline model. Provides clear probabilities but assumes linearity.', 'Simple concept but computationally expensive for prediction and memory-heavy.', 'Very powerful and effective, especially for non-linear problems, but requires careful tuning.', 'Can model highly complex relationships but is a "black box" and needs extensive data/tuning.', 'Often the best out-of-the-box performer. Robust, accurate, and provides feature importance.']
    }
    qualitative_df = pd.DataFrame(qualitative_data).set_index("Algorithm")
    final_comparison_df = qualitative_df.join(metrics_df)

    # --- Display Results in Tabs ---
    st.header("📊 Algorithm Comparison")
    
    tab_list = ["Overview"] + list(models.keys())
    tabs = st.tabs(tab_list)

    # Overview Tab
    with tabs[0]:
        st.subheader("Comparison Summary")
        # Create a display-only dataframe with formatted numbers
        display_df = final_comparison_df.copy()
        for col in ['Precision', 'Recall', 'F1-Score', 'AUC-ROC']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")
        display_df['Training Time (s)'] = display_df['Training Time (s)'].apply(lambda x: f"{x:.3f}")
        st.dataframe(display_df, use_container_width=True)

    # Individual Algorithm Tabs
    for i, algorithm_name in enumerate(models.keys()):
        with tabs[i+1]:
            st.subheader(f"Details for: {algorithm_name}")
            algo_data = final_comparison_df.loc[algorithm_name]

            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Precision", f"{algo_data['Precision']:.3f}")
                st.metric("Recall", f"{algo_data['Recall']:.3f}")
            with col2:
                st.metric("F1-Score", f"{algo_data['F1-Score']:.3f}")
                st.metric("AUC-ROC", f"{algo_data['AUC-ROC']:.3f}")
            with col3:
                 st.metric("Training Time (s)", f"{algo_data['Training Time (s)']:.3f}")

            st.markdown("---")
            
            # Display qualitative analysis
            st.subheader("Qualitative Analysis")
            qualitative_display_df = algo_data.drop(['Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Training Time (s)']).to_frame(name="Description")
            st.table(qualitative_display_df)

    # --- Expander for Raw Data ---
    with st.expander("View Raw Dataset"):
        st.dataframe(df_original, use_container_width=True)

if __name__ == '__main__':
    main()

