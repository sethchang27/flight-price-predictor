import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from feature_engineering_encoder import FeatureEngineeringEncoder
from feature_selector_encoder import FeatureSelector
from target_encoder import TargetEncoder
from one_hot_encoder import OneHotEncoder 


def create_flight_price_pipeline():
    """
    Creates the complete flight price prediction pipeline using custom encoders.
    
    Pipeline Steps:
    1. Feature Engineering - creates new features, log transforms target
    2. Feature Selection - selects relevant features based on EDA
    3. One-Hot Encoding - encodes low-cardinality categorical features  
    4. Target Encoding - encodes high-cardinality categorical features
    5. Model - GradientBoostingRegressor with optimized hyperparameters
    """

    pipeline = Pipeline([
        ('feature_engineering', FeatureEngineeringEncoder()),
        ('feature_selector', FeatureSelector()),
        ('one_hot_encoder', OneHotEncoder()),
        ('target_encoder', TargetEncoder()),
        ('model', GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
            )
        )
    ])

    return pipeline


def load_data(file_path='/Users/changseth/Desktop/ML Projects/Flight Price Predictor/Notebooks/flight_dataset.csv'):
    print('Loading dataset...')

    X = pd.read_csv(file_path)
    X['log_price'] = np.log1p(X['Price'])
    X = X.drop(columns=['Price'])
    y = X['log_price']

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {list(X.columns)}")

    return X, y


def train_and_evaluate_pipeline(X, y, test_size=0.2, random_state=42):
    """
    Train and evaluate the complete pipeline.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features dataset
    y : pd.Series
        Target variable
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random state for reproducible results
        
    Returns:
    --------
    pipeline : sklearn.pipeline.Pipeline
        Trained pipeline
    results : dict
        Evaluation results
    """

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


    # Train pipeline object 
    pipeline = create_flight_price_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    # Evaluate performance
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_r2 = r2_score(y_train, y_pred_train)

    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)

    # Print results
    print("\n" + "="*50)
    print("PIPELINE EVALUATION RESULTS")
    print("="*50)

    print(f"Training RMSE:   {train_rmse:.4f}")
    print(f"Training MAE:    {train_mae:.4f}")
    print(f"Training r_2:     {train_r2:.4f}")
    print(f"Test RMSE:       {test_rmse:.4f}")
    print(f"Test MAE:        {test_mae:.4f}")
    print(f"Test r_2:         {test_r2:.4f}")

    results = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2
    }
    
    return pipeline, results


def cross_validate_pipeline(X, y, cv_folds=5):
    """
    Perform cross-validation on the pipeline.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features dataset
    y : pd.Series
        Target variable
    cv_folds : int
        Number of cross-validation folds
        
    Returns:
    --------
    cv_scores : np.array
        Cross-validation RMSE scores
    """
    
    print("\n" + "="*50)
    print("CROSS-VALIDATION")
    print("="*50)
    
    pipeline = create_flight_price_pipeline()
    
    # Perform cross-validation
    cv_scores = cross_val_score(
        pipeline, X, y,
        cv=cv_folds,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    # Convert to RMSE
    cv_rmse_scores = np.sqrt(-cv_scores)
    
    print(f"Cross-Validation Results ({cv_folds} folds):")
    print(f"RMSE scores: {cv_rmse_scores}")
    print(f"Mean RMSE:   {cv_rmse_scores.mean():.4f}")
    print(f"Std RMSE:    {cv_rmse_scores.std():.4f}")
    
    return cv_rmse_scores


def save_pipeline(pipeline, filepath='trained_flight_price_pipeline.pkl'):
    """
    Save the trained pipeline to disk.
    
    Parameters:
    -----------
    pipeline : sklearn.pipeline.Pipeline
        Trained pipeline to save
    filepath : str
        Path where to save the pipeline
    """
    
    import joblib
    joblib.dump(pipeline, filepath)
    print("Pipeline saved successfully!")


def load_pipeline(filepath='trained_flight_price_pipeline.pkl'):
    """
    Load a trained pipeline from disk.
    
    Parameters:
    -----------
    filepath : str
        Path to the saved pipeline
        
    Returns:
    --------
    pipeline : sklearn.pipeline.Pipeline
        Loaded pipeline
    """
    
    import joblib
    pipeline = joblib.load(filepath)
    print("Pipeline loaded successfully!")

    return pipeline


def main():
    """
    Main function to run the complete flight price prediction pipeline.
    """
    
    try:
        X, y = load_data()
        pipeline, results = train_and_evaluate_pipeline(X, y)
        cv_scores = cross_validate_pipeline(X, y)
        save_pipeline(pipeline)
        
        print("\nPipeline training and evaluation completed successfully!")
        print(f"Final Test RMSE: {results['test_rmse']:.2f}")
        print(f"Final Test R²: {results['test_r2']:.4f}")
        
        return pipeline, results
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the complete pipeline
    trained_pipeline, evaluation_results = main()
