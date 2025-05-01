from flask import jsonify
import random

def register_tracking_route(app):
    @app.route('/tracking/realtime', methods=['GET'])
    def realtime_tracking():
        data = {
            "latitude": 5.3480 + random.uniform(-0.01, 0.01),
            "longitude": -4.0037 + random.uniform(-0.01, 0.01),
            "speed": round(random.uniform(20, 80), 2),
            "humidity": round(random.uniform(50, 100), 1),
            "congestion": random.choice(["Faible", "Moyenne", "Élevée"]),
            "incident": random.choice([True, False]),
            "type_incident": random.choice(["", "Accident", "Panne", "Travaux"])
        }
        return jsonify(data)
