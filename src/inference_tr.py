import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable

# Define the PositionalEncoding layer again for loading the model
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

class ElectricityDemandPredictor:
    def __init__(self, model, scaler=None):
        """
        Initialize the predictor with a trained model and scaler
        
        Parameters:
        -----------
        model_path : str
            Path to the saved Transformer model
        scaler_path : str, optional
            Path to the saved scaler. If None, you'll need to fit a new scaler
        """
        # Load the saved model with custom objects
        # self.model = load_model(model_path, custom_objects={'PositionalEncoding': PositionalEncoding})
        self.time_steps = 24  # Same as training
        
        # # Load or initialize scaler
        # if scaler_path and os.path.exists(scaler_path):
        #     import joblib
        #     self.scaler = joblib.load(scaler_path)
        # else:
        #     self.scaler = None
        #     print("Warning: No scaler provided. You'll need to fit one with fit_scaler()")
        self.model = model
        self.scaler = scaler
    
    def fit_scaler(self, historical_data):
        """
        Fit a new scaler on historical data
        
        Parameters:
        -----------
        historical_data : pd.DataFrame or np.array
            Historical demand data to fit the scaler
        """
        self.scaler = MinMaxScaler(feature_range=(0,1))
        if isinstance(historical_data, pd.DataFrame):
            self.scaler.fit(historical_data[['demand']])
        else:
            self.scaler.fit(historical_data.reshape(-1, 1))
        
        # Optionally save the scaler
        import joblib
        joblib.dump(self.scaler, 'electricity_demand_scaler.joblib')
        print("Scaler fitted and saved as 'electricity_demand_scaler.joblib'")
            
    def create_sequence(self, data):
        """
        Create a sequence from the last time_steps values
        
        Parameters:
        -----------
        data : np.array
            Array of demand values
        
        Returns:
        --------
        np.array
            Shaped input for model prediction
        """
        # Use the last time_steps values
        sequence = data[-self.time_steps:]
        
        # Reshape for the model
        return np.reshape(sequence, (1, self.time_steps, 1))
    
    def predict_next(self, recent_data):
        """
        Predict the next value given recent data
        
        Parameters:
        -----------
        recent_data : pd.DataFrame or np.array
            Recent demand data (at least time_steps points)
            
        Returns:
        --------
        float
            Predicted demand value
        """
        if self.scaler is None:
            raise ValueError("Scaler not initialized. Call fit_scaler first.")
            
        # Scale the input data
        if isinstance(recent_data, pd.DataFrame):
            scaled_data = self.scaler.transform(recent_data[['demand']].values)
        else:
            scaled_data = self.scaler.transform(recent_data.reshape(-1, 1))
        
        # Create sequence for prediction
        X = self.create_sequence(scaled_data)
        
        # Make prediction
        scaled_prediction = self.model.predict(X)[0][0]
        
        # Inverse transform to get actual value
        prediction = self.scaler.inverse_transform([[scaled_prediction]])[0][0]
        
        return prediction
    
    def predict_next_n_steps(self, recent_data, n_steps=24):
        """
        Predict multiple steps ahead
        
        Parameters:
        -----------
        recent_data : pd.DataFrame or np.array
            Recent demand data (at least time_steps points)
        n_steps : int
            Number of future steps to predict
            
        Returns:
        --------
        np.array
            Array of predicted values
        """
        if self.scaler is None:
            raise ValueError("Scaler not initialized. Call fit_scaler first.")
            
        # Initialize input data
        if isinstance(recent_data, pd.DataFrame):
            working_data = self.scaler.transform(recent_data[['demand']].values)
        else:
            working_data = self.scaler.transform(recent_data.reshape(-1, 1))
            
        working_data = working_data.flatten()
        
        # Store predictions
        predictions = []
        
        # Predict n steps ahead
        for _ in range(n_steps):
            # Use the last time_steps points for prediction
            sequence = working_data[-self.time_steps:]
            X = np.reshape(sequence, (1, self.time_steps, 1))
            
            # Make prediction
            next_scaled_value = self.model.predict(X, verbose=0)[0][0]
            
            # Add to working data for next iteration
            working_data = np.append(working_data, next_scaled_value)
            
            # Store prediction after inverse scaling
            prediction = self.scaler.inverse_transform([[next_scaled_value]])[0][0]
            predictions.append(prediction)
            predictions_df = pd.DataFrame(predictions, columns=['prediction'])
            
        return predictions_df
    
    def plot_prediction(self, historical_data, predictions, future_dates=None):
        """
        Plot historical data and predictions
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            Historical demand data with datetime index
        predictions : np.array
            Array of predicted values
        future_dates : pd.DatetimeIndex, optional
            Dates for predictions. If None, will be created.
        """
        plt.figure(figsize=(15, 6))
        
        # Plot historical data
        plt.plot(historical_data.index, historical_data['demand'], 
                 label='Historical Data', color='blue')
        
        # Create future dates if not provided
        if future_dates is None:
            last_date = historical_data.index[-1]
            freq = pd.infer_freq(historical_data.index)
            future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), 
                                         periods=len(predictions), freq=freq)
        
        # Plot predictions
        plt.plot(future_dates, predictions, 
                 label='Predictions', color='red', linestyle='--')
        
        plt.title('Electricity Demand: Historical vs Predicted')
        plt.xlabel('Date')
        plt.ylabel('Demand')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Load your test data
    # This would typically come from your database or files
    # test_data = pd.read_csv('electricity_test_data.csv', parse_dates=['date'], index_col='date')
    
    # Sample simulated data for demonstration
    dates = pd.date_range(start='2023-01-01', periods=100, freq='h')
    demand = np.sin(np.linspace(0, 8*np.pi, 100)) * 10 + 50 + np.random.normal(0, 1, 100)
    test_data = pd.DataFrame({'demand': demand}, index=dates)
    
    # Initialize predictor
    predictor = ElectricityDemandPredictor(
        model_path='models/transformer_model.keras',
        # scaler_path='electricity_demand_scaler.joblib'  # Uncomment if you have saved the scaler
    )
    
    # If no scaler was provided, fit one on historical data
    predictor.fit_scaler(test_data)
    
    # Predict next 24 hours
    predictions = predictor.predict_next_n_steps(test_data, n_steps=24)
    
    # Show predictions
    print("Predictions for next 24 hours:")
    for i, pred in enumerate(predictions):
        print(f"Hour {i+1}: {pred:.2f}")
    
    # Plot results
    predictor.plot_prediction(test_data, predictions)