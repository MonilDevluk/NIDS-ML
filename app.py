import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)
app.secret_key = "secret_key"  # required to store file path

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    result = session.get("result")
    message = session.get("message")
    session["message"] = None

    return render_template(
        "index.html",
        result=result,
        message=message
    )

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")

    if not file or file.filename == "":
        session["message"] = "Please upload a valid CSV file."
        return redirect(url_for("index"))

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    session["uploaded_file"] = filepath
    session["message"] = "File uploaded successfully!"
    return redirect(url_for("index"))

@app.route("/predict", methods=["POST"])
def predict():
    filepath = session.get("uploaded_file")

    if not filepath or not os.path.exists(filepath):
        session["message"] = "Upload a file before prediction!"
        return redirect(url_for("index"))

    df = pd.read_csv(filepath)

    pipeline = PredictPipeline()
    preds = pipeline.predict(df)

    counts = pd.Series(preds).value_counts().to_dict()

    session["result"] = counts
    session["message"] = "Prediction completed!"
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
