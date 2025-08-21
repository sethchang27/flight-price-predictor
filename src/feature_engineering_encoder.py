import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime

class FeatureEngineeringEncoder(BaseEstimator, TransformerMixin):
    """
    Feature engineering transformer for flight price prediction.
    """

    def __init__(self):
        # List of holidays in India for the year 2019
        self.india_holidays_2019 = [
            '26.01.2019', '04.03.2019', '21.03.2019', '17.04.2019', '18.05.2019',
            '05.06.2019', '12.08.2019', '15.08.2019', '19.08.2019', '24.08.2019',
            '10.09.2019', '02.10.2019', '08.10.2019', '27.10.2019', '10.11.2019',
            '12.11.2019', '25.12.2019'
        ]

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.copy()

        # Create date and day name features
        X_new = self._create_date_features(X_new)

        # Create cyclical day encoding
        X_new = self._create_cyclical_day_encoding(X_new)

        # Create time of day categories
        X_new = self._create_time_of_day_categories(X_new)

        # Create holiday indicator
        X_new = self._create_holiday_indicator(X_new)

        # Create duration bin features
        X_new = self._create_duration_features(X_new)

        # Drop unnecessary columns for modelling
        X_new = self._drop_unnecessary_columns(X_new)

        return X_new

    # Helper functions to create new features
    def _create_date_features(self, X):
        X['date'] = pd.to_datetime({
            'year': X['Year'],
            'month': X['Month'],
            'day': X['Day']
        })
    
        X['Day_Name'] = X['date'].dt.day_name()
        return X
    

    def _create_cyclical_day_encoding(self, X):
        X['Day_Num'] = X['date'].dt.dayofweek
        X['Day_sin'] = np.sin(2 * np.pi * X['Day_Num'] / 7)
        X['Day_cos'] = np.cos(2 * np.pi * X['Day_Num'] / 7)
        X = X.drop('Day_Num', axis=1)
        return X
    

    def _create_time_of_day_categories(self, X):
        X['dep_time_of_day'] = X['Dep_hours'].apply(self._time_of_day)
        X['arr_time_of_day'] = X['Arrival_hours'].apply(self._time_of_day)
        return X
    

    def _create_holiday_indicator(self, X):
        X['is_holiday'] = X['date'].apply(self._is_holiday)
        return X


    def _create_duration_features(self, X):
        bins = [0, 3, 6, 9, 12, float('inf')]
        duration_bins = ['1-3 Hours', '4-6 Hours', '7-9 Hours', '10-12 Hours', '12+ Hours']
        X['duration_category'] = pd.cut(
            X['Duration_hours'], 
            bins=bins, 
            labels=duration_bins
        )
        
        # Total duration in hours
        X['duration_total_hours'] = (
            X['Duration_hours'] + X['Duration_min'] / 60
        ).round(2)
        
        return X
    
    # Drop columns that are not needed for modeling
    def _drop_unnecessary_columns(self, X):
        columns_to_drop = [
            'Day', 'Year', 'date', 'Dep_hours', 'Dep_min', 'Arrival_hours', 'Arrival_min', 'Duration_hours', 'Duration_min', 'Day_Num'
        ]

        return X.drop(columns=columns_to_drop, errors='ignore')
    
    # Helper function for _create_time_of_day_categories()
    def _time_of_day(self, hour):
        if 5 <= hour <= 11:
            return 'Morning'
        elif 12 <= hour <= 15:
            return 'Afternoon'
        elif 16 <= hour <= 18:
            return 'Evening'
        else:
            return 'Night'
    
    # Helper function for _create_holiday_indicator()
    def _is_holiday(self, date):
        dates = pd.to_datetime(self.india_holidays_2019, format='%d.%m.%Y')
        
        # Window of +- 1 day around each public holiday
        time_frames = [(d - pd.Timedelta(days=1), d + pd.Timedelta(days=1)) for d in dates]
        
        for start, end in time_frames:
            if start <= date <= end:
                return 1
        return 0