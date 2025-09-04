from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            age=int(request.form.get('age')),
            cgpa=float(request.form.get('cgpa')),
            depression=int(request.form.get('depression')),
            anxiety=int(request.form.get('anxiety')),
            panic_attack=int(request.form.get('panic_attack')),
            specialist_treatment=int(request.form.get('specialist_treatment')),
            symptom_frequency_last7days=int(request.form.get('symptom_frequency_last7days')),
            has_mental_health_support=int(request.form.get('has_mental_health_support')),
            sleep_quality=int(request.form.get('sleep_quality')),
            study_stress_level=int(request.form.get('study_stress_level')),
            study_hours_per_week=int(request.form.get('study_hours_per_week')),
            academic_engagement=int(request.form.get('academic_engagement'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        pred_int = int(results[0])
        pred_label = "Counseling Needed" if pred_int == 1 else "No Counseling Needed"
        return render_template('home.html',results=pred_label)
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        