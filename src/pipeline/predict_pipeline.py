import sys
import os
import pandas as pd
from src.exception import CustomeException
from src.utils.filesManager import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomeException(e,sys)



class CustomData:
    def __init__(  self,
        gender: str,
        age: int,
        cgpa: float,
        depression: int,
        anxiety: int,
        panic_attack: int,
        specialist_treatment: int,
        symptom_frequency_last7days: int,
        has_mental_health_support: int,
        sleep_quality: int,
        study_stress_level: int,
        study_hours_per_week: int,
        academic_engagement: int):

        self.gender = gender
        self.age = age
        self.cgpa = cgpa
        self.depression = depression
        self.anxiety = anxiety
        self.panic_attack = panic_attack
        self.specialist_treatment = specialist_treatment
        self.symptom_frequency_last7days = symptom_frequency_last7days
        self.has_mental_health_support = has_mental_health_support
        self.sleep_quality = sleep_quality
        self.study_stress_level = study_stress_level
        self.study_hours_per_week = study_hours_per_week
        self.academic_engagement = academic_engagement

    def get_data_as_data_frame(self):
        try:
            # Normalize categorical values to match training capitalization
            gender_norm = str(self.gender).strip().title()

            custom_data_input_dict = {
                "Gender": [gender_norm],
                "Age": [self.age],
                "CGPA": [self.cgpa],
                "Depression": [self.depression],
                "Anxiety": [self.anxiety],
                "PanicAttack": [self.panic_attack],
                "SpecialistTreatment": [self.specialist_treatment],
                "SymptomFrequency_Last7Days": [self.symptom_frequency_last7days],
                "HasMentalHealthSupport": [self.has_mental_health_support],
                "SleepQuality": [self.sleep_quality],
                "StudyStressLevel": [self.study_stress_level],
                "StudyHoursPerWeek": [self.study_hours_per_week],
                "AcademicEngagement": [self.academic_engagement],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomeException(e, sys)