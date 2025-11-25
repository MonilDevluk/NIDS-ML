import os 
import pandas as pd 
from flask import Flask, render_template, request, redirect, url_for
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # User uploads csv
        file = request.files.get("file")
        if file is None or file.filename == "":
            return render_template("index.html", error="Please upload a valid CSV file.")
        
        try:
            # Read CSV directly from upload stream
            df = pd.read_csv(file)
            
            pipeline = PredictPipeline()
            preds = pipeline.predict(df)
            
            
            # Attach prediction back to original df
            result_df = df.copy()
            result_df['Prediction'] = preds
            
            # Calculate counts
            counts = result_df['Prediction'].value_counts().to_dict()
            
            # Show only first 50 rows in table 
            preview = result_df.head(50).to_dict(orient='records')
            
            return render_template(
                "index.html",
                predictions_preview=preview,
                counts=counts,
                success="Prediction completed successfully!"
            )
            
        except Exception as e:
            return render_template("index.html", error=f"Error: {str(e)}")
        
    # GET request
    return render_template("index.html")
if __name__ == '__main__':
    # Degug = True for development only
    app.run(debug=True)
    
    