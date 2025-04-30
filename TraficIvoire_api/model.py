import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import joblib
import os

class TrafficModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self, csv_path):
        """Load and preprocess traffic data from CSV"""
        df = pd.read_csv(csv_path)
        
        # Convert timestamp to features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_workday'] = df['day_of_week'].apply(lambda x: 1 if x < 5 else 0)
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        features = df[['latitude', 'longitude', 'hour', 'day_of_week', 'is_workday']].values
        labels = df['congestion_level'].values
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        return features_scaled, labels
    
    def train(self, features, labels):
        """Train the deep learning model"""
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        self.model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(5,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        return history, self.model.evaluate(X_test, y_test)
    
    def predict(self, features):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)
    
    def save(self, model_path='models'):
        """Save the model and scaler"""
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
        self.model.save(os.path.join(model_path, 'traffic_model.h5'))
        joblib.dump(self.scaler, os.path.join(model_path, 'scaler.pkl'))
    
    def load(self, model_path='models'):
        """Load the saved model and scaler"""
        self.model = keras.models.load_model(os.path.join(model_path, 'traffic_model.h5'))
        self.scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))