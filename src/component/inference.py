from datetime import datetime, timedelta

import hopsworks
# from hsfs.feature_store import FeatureStore
import pandas as pd
import numpy as np

import src.component.feature_group_config as config
from src.component.feature_store_api import get_feature_store, get_or_create_feature_view
from src.component.feature_group_config import FEATURE_VIEW_METADATA

def get_hopsworks_project() -> hopsworks.project.Project:

    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )

import pandas as pd

# def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
#     """Generate model predictions based on input features."""
    
#     assert features.shape[0] > 0, "Make sure your feature pipeline is up and running"
    
#     predictions = model.predict(features)

#     results = pd.DataFrame()
#     results['sub_region_code'] = features['sub_region_code'].values
#     results['predicted_demand'] = predictions.round(0)
    
#     return results  # Ensure the function returns the DataFrame

def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    """Generate model predictions based on input features."""
    
    assert features.shape[0] > 0, "Make sure your feature pipeline is up and running"
    
    # Extract sub_region_code before any processing
    sub_region_codes = features['sub_region_code'].values
    
    # For a transformer model expecting (batch, 24, 1) shape, we need to extract 24 time steps
    # Let's identify and extract the time series columns (assuming they follow a pattern)
    num_samples = features.shape[0]
    
    # Option 1: Find demand_previous_X_hour columns and sort them
    demand_cols = [col for col in features.columns if 'demand_previous' in col and 'hour' in col]
    if len(demand_cols) >= 24:
        # Sort columns by hour (assuming format like demand_previous_X_hour)
        demand_cols = sorted(demand_cols, 
                             key=lambda x: int(x.split('_')[2]) if x.split('_')[2].isdigit() else 0,
                             reverse=True)
        # Take the 24 most recent hours
        demand_cols = demand_cols[:24]
        
        # Create the properly shaped input array (samples, 24, 1)
        model_input = np.zeros((num_samples, 24, 1))
        for i in range(num_samples):
            for j, col in enumerate(demand_cols):
                model_input[i, j, 0] = features.iloc[i][col]
    else:
        # Option 2: Just reshape the numerical columns
        print(f"Warning: Could not find 24 demand columns. Using generic reshaping.")
        # Drop non-numeric columns to avoid errors
        numeric_features = features.select_dtypes(include=['number'])
        # If there are exactly 24 columns, reshape directly
        if numeric_features.shape[1] == 24:
            model_input = numeric_features.values.reshape(num_samples, 24, 1)
        else:
            # Otherwise, we need to pick or aggregate columns
            print(f"Error: Cannot automatically determine how to reshape {numeric_features.shape[1]} columns into 24 time steps.")
            # As a last resort, you could pick the first 24 columns or aggregate them
            if numeric_features.shape[1] > 24:
                model_input = numeric_features.iloc[:, :24].values.reshape(num_samples, 24, 1)
            else:
                raise ValueError(f"Not enough numeric columns ({numeric_features.shape[1]}) to create 24 time steps")
    
    # Make predictions
    predictions = model.predict(model_input)
    
    # Create results DataFrame
    results = pd.DataFrame()
    results['sub_region_code'] = sub_region_codes
    results['predicted_demand'] = predictions.flatten().round(0)
    
    return results




from datetime import timedelta

def load_batch_of_features_from_store(current_date: pd.Timestamp) -> pd.DataFrame:
    """Fetches the batch of features used by the ML system at `current_date`

    Args:
        current_date (datetime): datetime of the prediction for which we want
        to get the batch of features

    Returns:
        pd.DataFrame: 4 columns:
            - `date`
            - `demand`
            - `sub_region_code`
            - `tempreture`
    """
    n_features = config.N_FEATURES
    feature_view = get_or_create_feature_view(FEATURE_VIEW_METADATA)

    # Define time range
    fetch_data_to = pd.to_datetime(current_date - timedelta(hours=1), utc=True)
    fetch_data_from = pd.to_datetime(current_date - timedelta(days=28), utc=True)

    # Fetch data
    ts_data = feature_view.get_batch_data(
        start_time=fetch_data_from - timedelta(days=1),
        end_time=fetch_data_to + timedelta(days=1)
    )

    print('Dates before filtering:')
    print(ts_data['date'].min(), ts_data['date'].max())

    # Convert timestamps to milliseconds
    ts_from = int(fetch_data_from.timestamp() * 1000)
    ts_to = int(fetch_data_to.timestamp() * 1000)

    # Filter data for the required time period
    #ts_data = ts_data[ts_data.seconds.between(ts_from, ts_to)]
    ts_data = ts_data.groupby('sub_region_code').tail(672)
    print(ts_data.groupby('sub_region_code').tail(10))
    print('Dates after filtering:')
    print(ts_data['date'].min(), ts_data['date'].max())

    # Sort data by location and time
    ts_data.sort_values(by=['sub_region_code', 'date'], inplace=True)

    # Count records per sub-region
    location_counts = ts_data.groupby('sub_region_code').size()
    
    # Identify valid sub-regions that meet the required record count
    valid_sub_regions = location_counts[location_counts == config.N_FEATURES].index
    print(valid_sub_regions)

    # Filter the dataset to retain only valid sub-regions
    ts_data = ts_data[ts_data['sub_region_code'].isin(valid_sub_regions)]
    
    print(f"Filtered out sub-regions that do not meet the required {config.N_FEATURES} records.")

    # Transpose time-series data into a feature vector for each `sub_region_code`
    location_ids = ts_data['sub_region_code'].unique()
    x = np.ndarray(shape=(len(location_ids), n_features), dtype=np.float32)

    temperature_values = []
    for i, location_id in enumerate(location_ids):
        ts_data_i = ts_data.loc[ts_data.sub_region_code == location_id, :]
        ts_data_i = ts_data_i.sort_values(by=['date'])
        x[i, :] = ts_data_i['demand'].values
        temperature_values.append(ts_data_i['temperature_2m'].iloc[-1])

    # Convert numpy arrays to Pandas dataframe
    features = pd.DataFrame(
        x,
        columns=[f'demand_previous_{i+1}_hour' for i in reversed(range(n_features))]
    )
    features['temperature_2m'] = temperature_values
    features['date'] = current_date
    features['sub_region_code'] = location_ids
    features.sort_values(by=['sub_region_code'], inplace=True)

    return features
    

# def load_model_from_registry():
    
#     import joblib
#     from pathlib import Path
#     from tensorflow.keras.models import load_model

#     project = get_hopsworks_project()
#     model_registry = project.get_model_registry()

#     model = model_registry.get_model(
#         name=config.MODEL_NAME,
#         version=config.MODEL_VERSION,
#     )  
    
    
#     model_dir = model.download()
#     # model = joblib.load(Path(model_dir)  / 'LGB_model.pkl')
#     model = load_model(Path(model_dir) / 'transformer_model.keras', custom_objects={'PositionalEncoding': PositionalEncoding})


       
#     return model

# def load_model_from_registry():
#     import joblib
#     import numpy as np
#     import tensorflow as tf
#     from pathlib import Path
#     from tensorflow.keras.models import load_model
#     from tensorflow.keras.utils import register_keras_serializable
    
#     @register_keras_serializable()
#     class PositionalEncoding(tf.keras.layers.Layer):
#         def __init__(self, position, d_model, **kwargs):
#             super(PositionalEncoding, self).__init__(**kwargs)
#             self.position = position
#             self.d_model = d_model
#             self.pos_encoding = self.positional_encoding(position, d_model)

#         def positional_encoding(self, position, d_model):
#             angle_rads = self.get_angles(
#                 np.arange(position)[:, np.newaxis],
#                 np.arange(d_model)[np.newaxis, :],
#                 d_model
#             )
#             # Apply sin to even indices in the array; cos to odd indices
#             angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
#             angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
#             return tf.cast(angle_rads[np.newaxis, ...], tf.float32)

#         def get_angles(self, pos, i, d_model):
#             angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
#             return pos * angle_rates

#         def call(self, inputs):
#             return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

#         def get_config(self):
#             config = super(PositionalEncoding, self).get_config()
#             config.update({
#                 'position': self.position,
#                 'd_model': self.d_model
#             })
#             return config
    
#     project = get_hopsworks_project()
#     model_registry = project.get_model_registry()
    
#     model = model_registry.get_model(
#         name=config.MODEL_NAME,
#         version=config.MODEL_VERSION,
#     )
    
#     model_dir = model.download()
#     model = load_model(
#         Path(model_dir) / 'transformer_model.keras', 
#         custom_objects={'PositionalEncoding': PositionalEncoding}
#     )
    
#     return model

def load_model_and_scaler_from_registry():
    import joblib
    import numpy as np
    import tensorflow as tf
    from pathlib import Path
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import register_keras_serializable
    
    @register_keras_serializable()
    class PositionalEncoding(tf.keras.layers.Layer):
        def __init__(self, position, d_model, **kwargs):
            super(PositionalEncoding, self).__init__(**kwargs)
            self.position = position
            self.d_model = d_model
            self.pos_encoding = self.positional_encoding(position, d_model)

        def positional_encoding(self, position, d_model):
            angle_rads = self.get_angles(
                np.arange(position)[:, np.newaxis],
                np.arange(d_model)[np.newaxis, :],
                d_model
            )
            # Apply sin to even indices in the array; cos to odd indices
            angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
            angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
            return tf.cast(angle_rads[np.newaxis, ...], tf.float32)

        def get_angles(self, pos, i, d_model):
            angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
            return pos * angle_rates

        def call(self, inputs):
            return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

        def get_config(self):
            config = super(PositionalEncoding, self).get_config()
            config.update({
                'position': self.position,
                'd_model': self.d_model
            })
            return config
    
    project = get_hopsworks_project()
    model_registry = project.get_model_registry()
    
    # Get the model from registry
    model_entry = model_registry.get_model(
        name=config.MODEL_NAME,
        version=config.MODEL_VERSION,
    )
    
    # Download the model artifacts
    model_dir = model_entry.download()
    
    # Load the Keras model with custom objects
    transformer_model = load_model(
        Path(model_dir) / 'transformer_model.keras', 
        custom_objects={'PositionalEncoding': PositionalEncoding}
    )
    
    # Load the scaler
    scaler = joblib.load(Path(model_dir) / 'minmax_scaler.pkl')
    
    return transformer_model, scaler

def load_predictions_from_store(
    from_date: datetime,
    to_date: datetime
    ) -> pd.DataFrame:
    """
    Connects to the feature store and retrieves model predictions for all
    `sub_region_code`s and for the time period from `from_date`
    to `to_date`

    Args:
        from_date (datetime): min datetime (rounded hour) for which we want to get
        predictions

        to_date(datetime): max datetime (rounded hour) for which we want to get
        predictions

    Returns:
        pd.DataFrame: 3 columns:
            - `sub_region_code`
            - `predicted_demand`
            - `date`
    """
    from src.component.feature_group_config import FEATURE_VIEW_PREDICTIONS_METADATA
    from src.component.feature_store_api import get_or_create_feature_view

    # get pointer to the feature view
    predictions_fv = get_or_create_feature_view(FEATURE_VIEW_PREDICTIONS_METADATA)

    # get data from the feature view
    print(f'Fetching predictions for `date` between {from_date}  and {to_date}')
    predictions = predictions_fv.get_batch_data(
        start_time=from_date - timedelta(days=1),
        end_time=to_date + timedelta(days=1)
    )
    
    # make sure datetimes are UTC aware
    predictions['date'] = pd.to_datetime(predictions['date'], utc=True)
    from_date = pd.to_datetime(from_date, utc=True)
    to_date = pd.to_datetime(to_date, utc=True)

    # make sure we keep only the range we want
    predictions = predictions[predictions.date.between(from_date, to_date)]

    # sort by `pick_up_hour` and `pickup_location_id`
    predictions.sort_values(by=['date', 'sub_region_code'], inplace=True)

    return predictions