import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.utils.filesManager import load_object
import os

def test_prediction_pipeline():
    """Test the prediction pipeline with sample data"""
    print("🧪 Testing Prediction Pipeline...")
    
    # Test 1: Check if artifacts exist
    print("\n1. Checking artifacts...")
    model_path = os.path.join("artifacts", "model.pkl")
    preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
    
    if not os.path.exists(model_path):
        print("❌ Model artifact not found!")
        return False
    if not os.path.exists(preprocessor_path):
        print("❌ Preprocessor artifact not found!")
        return False
    print("✅ Artifacts found")
    
    # Test 2: Load model and preprocessor
    print("\n2. Loading model and preprocessor...")
    try:
        model = load_object(model_path)
        preprocessor = load_object(preprocessor_path)
        print("✅ Model and preprocessor loaded successfully")
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        return False
    
    # Test 3: Create sample data
    print("\n3. Creating sample data...")
    try:
        sample_data = CustomData(
            gender="Male",
            age=20,
            cgpa=3.5,
            depression=0,
            anxiety=0,
            panic_attack=0,
            specialist_treatment=0,
            symptom_frequency_last7days=1,
            has_mental_health_support=1,
            sleep_quality=4,
            study_stress_level=2,
            study_hours_per_week=15,
            academic_engagement=4
        )
        df = sample_data.get_data_as_data_frame()
        print("✅ Sample data created")
        print(f"   Sample data shape: {df.shape}")
        print(f"   Sample data columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        return False
    
    # Test 4: Test prediction pipeline
    print("\n4. Testing prediction pipeline...")
    try:
        pipeline = PredictPipeline()
        prediction = pipeline.predict(df)
        print("✅ Prediction successful")
        print(f"   Raw prediction: {prediction}")
        print(f"   Prediction type: {type(prediction)}")
        print(f"   Prediction shape: {prediction.shape if hasattr(prediction, 'shape') else 'scalar'}")
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return False
    
    # Test 5: Test multiple scenarios
    print("\n5. Testing multiple scenarios...")
    scenarios = [
        {
            "name": "Low Risk",
            "data": {
                "gender": "Female",
                "age": 19,
                "cgpa": 3.8,
                "depression": 0,
                "anxiety": 0,
                "panic_attack": 0,
                "specialist_treatment": 0,
                "symptom_frequency_last7days": 0,
                "has_mental_health_support": 1,
                "sleep_quality": 5,
                "study_stress_level": 1,
                "study_hours_per_week": 22,
                "academic_engagement": 5
            }
        },
        {
            "name": "High Risk",
            "data": {
                "gender": "Male",
                "age": 22,
                "cgpa": 2.0,
                "depression": 1,
                "anxiety": 1,
                "panic_attack": 1,
                "specialist_treatment": 1,
                "symptom_frequency_last7days": 7,
                "has_mental_health_support": 0,
                "sleep_quality": 1,
                "study_stress_level": 5,
                "study_hours_per_week": 5,
                "academic_engagement": 1
            }
        }
    ]
    
    for scenario in scenarios:
        try:
            test_data = CustomData(**scenario["data"])
            test_df = test_data.get_data_as_data_frame()
            test_pred = pipeline.predict(test_df)
            pred_label = "Counseling Needed" if int(test_pred[0]) == 1 else "No Counseling Needed"
            print(f"   {scenario['name']}: {pred_label}")
        except Exception as e:
            print(f"   ❌ {scenario['name']} failed: {e}")
    
    # Test 6: Test data validation
    print("\n6. Testing data validation...")
    try:
        # Test with different gender formats
        test_gender = CustomData(
            gender="female",  # lowercase
            age=20,
            cgpa=3.0,
            depression=0,
            anxiety=0,
            panic_attack=0,
            specialist_treatment=0,
            symptom_frequency_last7days=2,
            has_mental_health_support=1,
            sleep_quality=3,
            study_stress_level=3,
            study_hours_per_week=12,
            academic_engagement=3
        )
        test_df = test_gender.get_data_as_data_frame()
        pred = pipeline.predict(test_df)
        print("✅ Gender normalization works")
    except Exception as e:
        print(f"❌ Gender normalization failed: {e}")
    
    print("\n🎉 All tests completed!")
    return True

def test_model_performance():
    """Test model performance on test data"""
    print("\n📊 Testing Model Performance...")
    
    try:
        # Load test data
        test_df = pd.read_csv("artifacts/test.csv")
        print(f"Test data shape: {test_df.shape}")
        
        # Separate features and target
        X_test = test_df.drop(columns=['IsCounselingNeeded'])
        y_test = test_df['IsCounselingNeeded']
        
        # Load model and preprocessor
        model = load_object("artifacts/model.pkl")
        preprocessor = load_object("artifacts/preprocessor.pkl")
        
        # Transform features
        X_test_transformed = preprocessor.transform(X_test)
        
        # Make predictions
        predictions = model.predict(X_test_transformed)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
        
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        
        print(f"✅ Model Performance:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        print(f"   Confusion Matrix:\n{cm}")
        print(f"   Classification Report:\n{classification_report(y_test, predictions)}")
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting Mental Health Prediction Pipeline Tests")
    print("=" * 50)
    
    # Run basic pipeline tests
    success = test_prediction_pipeline()
    
    if success:
        # Run performance tests
        test_model_performance()
    
    print("\n" + "=" * 50)
    print("🏁 Testing completed!")
