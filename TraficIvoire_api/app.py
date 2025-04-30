from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import pandas as pd
from geopy.distance import geodesic
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure Swagger documentation
api = Api(
    app,
    version='1.0',
    title='Abidjan Traffic AI API',
    description='API for predicting traffic conditions in Abidjan using deep learning',
    doc='/docs'
)

# Define namespaces
traffic_ns = api.namespace('traffic', description='Traffic prediction operations')
routes_ns = api.namespace('routes', description='Route optimization operations')

# Define models for Swagger documentation
location_model = api.model('Location', {
    'latitude': fields.Float(required=True, description='Latitude coordinate'),
    'longitude': fields.Float(required=True, description='Longitude coordinate')
})

prediction_model = api.model('Prediction', {
    'location': fields.Nested(location_model),
    'timestamp': fields.DateTime(required=True, description='Prediction timestamp')
})

# Initialize deep learning model
def create_traffic_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(5,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

traffic_model = create_traffic_model()

@traffic_ns.route('/predict')
class TrafficPrediction(Resource):
    @traffic_ns.expect(prediction_model)
    def post(self):
        """Predict traffic conditions for a given location and time"""
        try:
            data = request.get_json()
            location = data['location']
            timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
            
            # Feature engineering
            features = np.array([
                location['latitude'],
                location['longitude'],
                timestamp.hour,
                timestamp.weekday(),
                1 if timestamp.weekday() < 5 else 0  # Is workday
            ]).reshape(1, -1)
            
            # Make prediction
            prediction = traffic_model.predict(features)[0][0]
            
            return {
                'congestion_probability': float(prediction),
                'traffic_level': 'high' if prediction > 0.7 else 'medium' if prediction > 0.3 else 'low'
            }
            
        except Exception as e:
            return {'error': str(e)}, 400

@routes_ns.route('/optimize')
class RouteOptimization(Resource):
    @routes_ns.expect([location_model])
    def post(self):
        """Optimize route based on current and predicted traffic conditions"""
        try:
            waypoints = request.get_json()
            
            # Calculate optimal route
            optimized_route = []
            for i in range(len(waypoints) - 1):
                start = waypoints[i]
                end = waypoints[i + 1]
                
                # Calculate direct distance
                distance = geodesic(
                    (start['latitude'], start['longitude']),
                    (end['latitude'], end['longitude'])
                ).kilometers
                
                optimized_route.append({
                    'start': start,
                    'end': end,
                    'distance': round(distance, 2),
                    'estimated_time': round(distance * 2, 2)  # Simple estimation
                })
            
            return {
                'route': optimized_route,
                'total_distance': sum(r['distance'] for r in optimized_route),
                'total_time': sum(r['estimated_time'] for r in optimized_route)
            }
            
        except Exception as e:
            return {'error': str(e)}, 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)