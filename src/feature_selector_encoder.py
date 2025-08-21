import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        Feature selector for flight price prediction.
        This encoder selects relevant features based on correlation analysis done during EDA.
        """

        # List of selected features based on EDA
        self.selected_features_ = [
            'Airline',
            'Source',
            'Destination',
            'Total_Stops',
            'Month',
            'Day_Name',
            'dep_time_of_day',
            'arr_time_of_day',
            'duration_total_hours'
        ]

        self.target_col = 'log_price'
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """
        Selects relevant features from the input DataFrame.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input DataFrame containing flight data.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with only the selected features.
        """
        X_new = X[self.selected_features_ + [self.target_col]].copy()
        return X_new