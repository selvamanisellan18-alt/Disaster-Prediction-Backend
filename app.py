from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from predict import predict_disaster
from datetime import datetime  # <-- PUDHUSA ADD PANNIRUKKOM

# Import the utility modules
try:
    from utils.weather_api import WeatherFusion
    from utils.severity_scoring import SeverityCalculator
    weather_fusion = WeatherFusion()
    severity_calculator = SeverityCalculator()
    print("✅ Environmental Data Fusion modules loaded.")
except ImportError as e:
    weather_fusion = None
    severity_calculator = None
    print(f"❌ Error loading utility modules: {e}")

app = Flask(__name__)

# --- CONFIGURATION ---
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "jfif"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# GLOBAL LIST TO STORE LOGS IN MEMORY
prediction_logs = []

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# --- ROUTES ---

@app.route("/", methods=["GET"])
def home():
    # Pass the logs even when the page loads for the first time
    return render_template("index.html", prediction=None, logs=prediction_logs)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return render_template("index.html", error="No image uploaded", logs=prediction_logs)

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", error="No file selected", logs=prediction_logs)

        if not allowed_file(file.filename):
            return render_template("index.html", error="Only JPG/PNG allowed", logs=prediction_logs)

        # Save the file safely
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Send to AI Engine
        result, confidence = predict_disaster(file_path)
        conf_percentage = round(confidence * 100, 2)

        # Fetch Environmental Data
        weather_data = None
        severity_info = None
        
        status_badge = "CRITICAL"
        
        if result == "INVALID INPUT":
            status_badge = "REJECTED"
        elif result in ["Normal", "Stable Terrain"]:
            status_badge = "SAFE"
        
        if result != "INVALID INPUT" and weather_fusion and severity_calculator:
            weather_data = weather_fusion.get_realtime_weather()
            if result in ["Cyclone", "Flood", "Earthquake", "Fire"]:
                severity_info = severity_calculator.calculate_score(65.0, 8500.0, 500.0)
            else:
                 severity_info = severity_calculator.calculate_score(0, 0, 0)

        # --- LOG RECORDING LOGIC ---
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event": result,
            "confidence": "99.99%" if result == "INVALID INPUT" else f"{conf_percentage}%",
            "status": status_badge
        }
        
        # Add the new log to the top of the list
        prediction_logs.insert(0, log_entry)
        
        # Keep only the last 20 logs so memory doesn't get full
        if len(prediction_logs) > 20:
            prediction_logs.pop()

        return render_template(
            "index.html",
            prediction=result,
            confidence=conf_percentage,
            img_path=f"/{file_path}",
            weather=weather_data,
            severity=severity_info,
            logs=prediction_logs  # Pass logs to HTML
        )

    except Exception as e:
        return render_template("index.html", error=f"Processing failed: {str(e)}", logs=prediction_logs)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)