# utils/weather_api.py
import random

class WeatherFusion:
    def __init__(self):
        pass

    def get_realtime_weather(self, lat=13.0827, lon=80.2707):
        # Simulated weather data for Chennai coordinates
        conditions = ["Cloudy", "Heavy Rain", "Clear", "Thunderstorm", "High Winds"]
        return {
            "temperature": round(random.uniform(25.0, 35.0), 1),
            "humidity": random.randint(60, 95),
            "wind_speed": round(random.uniform(10.0, 60.0), 1),
            "condition": random.choice(conditions)
        }