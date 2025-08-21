import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder

class OneHotEncoder(BaseEstimator, TransformerMixin):
    """
    One-hot encode low-cardinality categorical features using sklearn's OneHotEncoder.
    """
    
    def __init__(self):
        # Features to one-hot encode
        self.categorical_features = [
            'dep_time_of_day', 
            'arr_time_of_day', 
            'Day_Name', 
            'Total_Stops', 
            'Month'
        ]
        
        self.encoders_ = {}
    
    def fit(self, X, y=None):
        self.encoders_ = {}
        
        for feature in self.categorical_features:
            # Create separate encoder for each feature
            encoder = SklearnOneHotEncoder(
                drop='first',  # Drop first category to avoid multicollinearity
                sparse_output=False, 
                handle_unknown='ignore'  
            )
            
            # Fit encoders on each feature
            encoder.fit(X[[feature]])
            self.encoders_[feature] = encoder
        
        return self
    
    def transform(self, X):
        X_new = X.copy()
        
        for feature in self.categorical_features:
            # Get encoder for this feature
            encoder = self.encoders_[feature]
            encoded_array = encoder.transform(X_new[[feature]])
            
            # Create new column names
            feature_names = [f"{feature}_{cat}" for cat in encoder.categories_[0][1:]]  

            # Create DataFrame with encoded features
            encoded_df = pd.DataFrame(
                encoded_array, 
                columns=feature_names,
                index=X_new.index
            )
            
            # Add encoded columns to dataframe
            X_new = pd.concat([X_new, encoded_df], axis=1)
            
            # Drop original categorical feature
            X_new = X_new.drop(columns=[feature])
        
        return X_new