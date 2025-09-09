import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib  # Used for saving the model

print("Starting model training process...")

# --- 1. DATA LOADING AND CLEANING (Slightly modified from original) ---
def load_and_clean_data(filepath):
    """Loads and cleans the life expectancy data."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Ensure 'year' is not imputed if it exists
    if 'year' in numeric_cols:
        numeric_cols.remove('year')

    for col in numeric_cols:
        # Group by country and fill missing values with the median for that country
        df[col] = df.groupby('country')[col].transform(lambda x: x.fillna(x.median()))
    
    # For any remaining NaNs (e.g., a country with all NaN values for a column), fill with the global median
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
        
    df.dropna(subset=['life_expectancy'], inplace=True)
    return df

# --- 2. MODEL TRAINING ---
def create_and_save_model(df):
    """Trains and saves the model, preprocessor, and feature names."""
    X = df.drop(['life_expectancy', 'country'], axis=1)
    y = df['life_expectancy']
    
    categorical_features = ['status']
    numerical_features = X.drop(categorical_features, axis=1).columns
    
    # Create the preprocessing pipeline for numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
        
    # Create the full model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    # Train the model on the ENTIRE dataset
    model_pipeline.fit(X, y)
    
    # --- FIXED SECTION ---
    # Get top features for the estimator form
    try:
        # Access the OneHotEncoder directly and get feature names
        ohe_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
        
        all_feature_names = np.concatenate([numerical_features, ohe_feature_names])
        importances = model_pipeline.named_steps['regressor'].feature_importances_
        
        feature_importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances}).sort_values('importance', ascending=False)
        print("\nFeature Importances:")
        print(feature_importance_df.head(10))

        # We will still use the original feature names for the form for simplicity
        raw_importances = feature_importance_df.copy()
        raw_importances['feature'] = raw_importances['feature'].apply(lambda x: x.split('_')[0] if 'status' not in x else 'status')
        top_features = raw_importances.groupby('feature')['importance'].sum().sort_values(ascending=False).head(7).index.tolist()

    except Exception as e:
        print(f"Could not extract feature names, using defaults: {e}")
        top_features = ['adult_mortality', 'income_composition_of_resources', 'hiv/aids', 'schooling', 'bmi', 'gdp', 'thinness_1_19_years']

    print(f"\nTop 7 features selected for the form: {top_features}")

    # Save the pipeline and top features to a file
    joblib.dump({
        'model': model_pipeline,
        'top_features': top_features
    }, 'life_expectancy_model.pkl')
    
    print("\nModel and supporting data saved to 'life_expectancy_model.pkl'")
    return top_features

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        data = load_and_clean_data("Life Expectancy Data.csv")
        create_and_save_model(data)
        print("\nProcess complete. You can now deploy your application.")
    except FileNotFoundError:
        print("\nERROR: 'Life Expectancy Data.csv' not found.")
        print("Please download the dataset from Kaggle and place it in the same folder.")

