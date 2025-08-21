import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoding transformer for flight price prediction.
    This transformer encodes categorical features based on the mean target value.
    """

    def __init__(self, 
                 smoothing=1.0,
                 min_samples_leaf=1,
                 noise_level=0.01,
                 random_state=42):
        """
        Parameters:
        -----------
        categorical_features : list, optional
            List of categorical column names to encode
        smoothing : float, default=1.0
            Smoothing factor for regularization (higher = more smoothing towards global mean)
        min_samples_leaf : int, default=1
            Minimum number of samples required to keep category encoding
        noise_level : float, default=0.01
            Small noise to add to encodings to prevent overfitting
        random_state : int, default=42
            Random state for noise generation
        """
        # Features to target encode
        self.categorical_features = [
            'Airline', 
            'Source', 
            'Destination'
        ]

        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.noise_level = noise_level
        self.random_state = random_state
        
        # Set during fit
        self.encoding_mappings_ = {}
        self.global_mean_ = None
        # self.feature_names_ = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        X = X.copy()

        # Calculate global mean for smoothing and unseen categories
        self.global_mean_ = y.mean()

        # Fit encoder for each categorical feature
        for feature in self.categorical_features:
            # Get number of samples per category
            category_stats = X.groupby(feature).agg({
                feature: 'count'
            }).rename(columns={feature: 'count'})

            # Mean price of each category
            target_means = X.groupby(feature).apply(lambda group_df: y[group_df.index].mean())

            # Map categories to their mean prices
            category_stats['target_mean'] = target_means

            # Apply smoothing (regularization towards global mean)
            # Allows for means of low sample categories to be closer to global mean to prevent overfitting
            category_stats['smoothed_mean'] = (
                (category_stats['count'] * category_stats['target_mean'] + 
                 self.smoothing * self.global_mean_) / 
                (category_stats['count'] + self.smoothing)
            )

            # Filter out categories with too few samples
            # Categories with fewer than min_samples_leaf will be encoded with global mean price
            valid_categories = category_stats[
                category_stats['count'] >= self.min_samples_leaf
            ]

            encoding_map = valid_categories['smoothed_mean'].to_dict()
            
            # Add small noise to encoded values prevent overfitting
            for cat in encoding_map:
                noise = np.random.normal(0, self.noise_level)
                encoding_map[cat] = encoding_map[cat] + noise
            
            # Store encoding mapping for the feature
            self.encoding_mappings_[feature] = encoding_map
        
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Apply encoding to each categorical feature
        for feature in self.categorical_features:
            # Map categories to their encoded values, using global mean for unseen categories
            X[feature + '_encoded'] = X[feature].map(
                self.encoding_mappings_[feature]
            ).fillna(self.global_mean_)
            
            # Drop original categorical feature
            X.drop(feature, axis=1, inplace=True)

        return X