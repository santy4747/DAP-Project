import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for server use
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

print("Starting model training process for Student Performance...")

# --- Helper function to convert plots to base64 strings ---
def fig_to_base64():
    """Converts a matplotlib figure to a base64 encoded string."""
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_str

# --- 1. DATA LOADING AND PREPARATION ---
def load_and_prep_data(filepath):
    """Loads and prepares the student performance data."""
    df = pd.read_csv(filepath, sep=',')
    
    features = [
        'Medu', 'Fedu', 'studytime', 'failures', 'schoolsup', 'famsup', 
        'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3'
    ]
    
    df = df[features]
    df.dropna(subset=['G3'], inplace=True)
    return df

# --- 2. EDA VISUALIZATION GENERATION ---
def generate_visualizations(df):
    """Generates key EDA plots and returns them as base64 strings."""
    print("Generating EDA visualizations...")
    visualizations = {}
    sns.set_style("whitegrid")

    # a. Distribution of Final Grades (G3)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['G3'], bins=20, kde=True, color='indigo')
    plt.title('Distribution of Final Student Grades (G3)', fontsize=16)
    plt.xlabel('Final Grade (out of 20)')
    plt.ylabel('Number of Students')
    visualizations['grade_distribution'] = fig_to_base64()

    # b. Failures vs Final Grade
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='failures', y='G3', data=df, palette='viridis')
    plt.title('Impact of Past Failures on Final Grade', fontsize=16)
    plt.xlabel('Number of Past Failures')
    plt.ylabel('Final Grade (G3)')
    visualizations['failures_impact'] = fig_to_base64()

    # c. Study Time vs Final Grade
    plt.figure(figsize=(10, 6))
    studytime_labels = {1: '< 2 hrs', 2: '2-5 hrs', 3: '5-10 hrs', 4: '> 10 hrs'}
    df['studytime_str'] = df['studytime'].map(studytime_labels)
    sns.boxplot(x='studytime_str', y='G3', data=df, order=studytime_labels.values(), palette='plasma')
    plt.title('Impact of Weekly Study Time on Final Grade', fontsize=16)
    plt.xlabel('Weekly Study Time')
    plt.ylabel('Final Grade (G3)')
    visualizations['studytime_impact'] = fig_to_base64()

    # d. Mother's Education ---
    plt.figure(figsize=(10, 6))
    medu_labels = {0: 'None', 1: 'Primary', 2: '5th-9th', 3: 'Secondary', 4: 'Higher Ed'}
    df['Medu_str'] = df['Medu'].map(medu_labels)
    sns.boxplot(x='Medu_str', y='G3', data=df, order=medu_labels.values(), palette='cool')
    plt.title("Mother's Education Level vs Final Grade", fontsize=16)
    plt.xlabel("Mother's Education")
    plt.ylabel('Final Grade (G3)')
    visualizations['medu_impact'] = fig_to_base64()

    # e. Alcohol Consumption ---
    plt.figure(figsize=(10, 6))
    df['total_alcohol'] = df['Dalc'] + df['Walc']
    sns.boxplot(x='total_alcohol', y='G3', data=df, palette='autumn')
    plt.title('Total Alcohol Consumption vs Final Grade', fontsize=16)
    plt.xlabel('Total Weekly Alcohol Consumption (Workday + Weekend, Scale 2-10)')
    plt.ylabel('Final Grade (G3)')
    visualizations['alcohol_impact'] = fig_to_base64()
    
    print("Visualizations generated successfully.")
    return visualizations

# --- 3. MODEL TRAINING ---
def create_and_save_model(df, visualizations):
    """Trains the model and saves it along with feature insights and EDA plots."""
    X = df.drop(['G3', 'studytime_str', 'Medu_str', 'total_alcohol'], axis=1, errors='ignore') 
    y = df['G3']
    
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ], remainder='passthrough')
        
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    model_pipeline.fit(X, y)
    
    try:
        ohe_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_feature_names = np.concatenate([numerical_features, ohe_feature_names])
        importances = model_pipeline.named_steps['regressor'].feature_importances_
        
        feature_importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances}).sort_values('importance', ascending=False)
        top_features = feature_importance_df['feature'].head(7).tolist()
    except Exception as e:
        print(f"Could not extract feature names, using defaults: {e}")
        feature_importance_df = None
        top_features = ['G2', 'G1', 'failures', 'absences', 'Medu', 'studytime']

    print(f"\nTop 7 features for the form: {top_features}")

    joblib.dump({
        'model': model_pipeline,
        'top_features': top_features,
        'feature_importances': feature_importance_df,
        'visualizations': visualizations
    }, 'student_performance_model.pkl')
    
    print("\nModel, insights, and visualizations saved to 'student_performance_model.pkl'")
    
    if feature_importance_df is not None:
        print_educational_interpretation(model_pipeline, df, feature_importance_df)
    
    return top_features

# --- 4. EDUCATIONAL INTERPRETATION FUNCTION ---
def print_educational_interpretation(model, df, feature_importances):
    print("\n" + "="*60)
    print("   EDUCATIONAL POLICY & INTERVENTION INSIGHTS")
    print("="*60)
    top_3_features = feature_importances[~feature_importances['feature'].isin(['G1', 'G2'])]['feature'].head(3).tolist()
    print("\nBeyond previous grades, the model identifies these key drivers for student success:\n")
    feature_1 = top_3_features[0]
    print(f"1. {feature_1.replace('_', ' ').title()}: The strongest predictor of future outcomes.")
    print("   Interpretation: The number of past class failures is a critical red flag.\n")
    feature_2 = top_3_features[1]
    print(f"2. {feature_2.replace('_', ' ').title()}: A key indicator of a student's environment.")
    print("   Interpretation: The mother's education level consistently appears as a top factor.\n")
    feature_3 = top_3_features[2]
    print(f"3. {feature_3.replace('_', ' ').title()}: A direct behavioral link to performance.")
    print("   Interpretation: The number of school absences is a direct and controllable predictor.\n")
    print("-" * 60)
    print("   INTERVENTION SIMULATION: What-If Scenario")
    print("-" * 60)
    
    # --- FIX START ---
    # Create a base DataFrame for simulation by dropping temporary and target columns
    df_for_sim = df.drop(['G3', 'studytime_str', 'Medu_str', 'total_alcohol'], axis=1, errors='ignore')

    # Create a baseline student profile using the median for numeric features 
    # and the mode (most common value) for categorical features.
    baseline_student = pd.DataFrame([
        {col: df_for_sim[col].median() if pd.api.types.is_numeric_dtype(df_for_sim[col]) else df_for_sim[col].mode()[0] 
         for col in df_for_sim.columns}
    ])

    # Create the 'at-risk' scenario by modifying the baseline
    at_risk = baseline_student.copy()
    at_risk['failures'] = 2
    at_risk['absences'] = 15
    at_risk['schoolsup'] = 'no' # This is a categorical feature
    # --- FIX END ---
    
    base_prediction = model.predict(at_risk)[0]
    print(f"An at-risk student has a predicted final grade of: {base_prediction:.2f}/20.")
    
    intervened_scenario = at_risk.copy()
    intervened_scenario['schoolsup'] = 'yes'
    intervened_scenario['absences'] = 5
    
    improved_prediction = model.predict(intervened_scenario)[0]
    gain = improved_prediction - base_prediction
    print(f"With intervention, the predicted grade is: {improved_prediction:.2f}/20 (a gain of {gain:.2f} points).\n")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        data = load_and_prep_data("student-mat.csv")
        eda_visualizations = generate_visualizations(data)
        create_and_save_model(data, eda_visualizations)
        print("\nProcess complete.")
    except FileNotFoundError:
        print("\nERROR: 'student-mat.csv' not found.")
        print("Please download the dataset and place it in the same folder.")
    except KeyError as e:
        print(f"\nCRITICAL ERROR: A column was not found. This is likely due to an incorrect CSV separator.")
        print(f"Please check if 'student-mat.csv' is comma-separated (,) or semicolon-separated (;).")
        print(f"Original error: {e}")

