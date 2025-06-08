from fastapi import FastAPI, HTTPException
from model_input import CovidPatientFeatures
import pickle as p
import pandas as pd
import numpy as np
from typing import Dict, Any

app = FastAPI(
    title="COVID Patient Hospitalization Predictor",
    description="API to predict if a COVID patient will be hospitalized",
    version="1.0.0"
)

# Load the model and related components
try:
    with open("random_forest_pipeline.pkl", "rb") as f:
        model_data = p.load(f)
    
    model = model_data['pipeline']
    label_encoder = model_data.get('label_encoder', None)
    feature_names = model_data.get('feature_names', [])
    numerical_columns = model_data.get('numerical_columns', [])
    categorical_columns = model_data.get('categorical_columns', [])
    
    print("Model loaded successfully!")
    print(f"Expected features: {feature_names}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.get("/")
def read_root():
    return {
        "message": "COVID Patient Hospitalization Prediction API",
        "status": "Model loaded" if model else "Model not loaded",
        "endpoints": {
            "predict": "/predict/",
            "health": "/health/",
            "model_info": "/model-info/"
        }
    }

@app.get("/health/")
def health_check():
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None
    }

@app.get("/model-info/")
def model_info():
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "Random Forest Classifier",
        "expected_features": feature_names,
        "numerical_features": numerical_columns,
        "categorical_features": categorical_columns,
        "has_label_encoder": label_encoder is not None,
        "target_classes": label_encoder.classes_.tolist() if label_encoder else ["0", "1"]
    }

@app.post("/predict/")
def predict(data: CovidPatientFeatures) -> Dict[str, Any]:
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to dictionary
        input_dict = {
            'Age': data.Age,
            'Gender': data.Gender,
            'Region': data.Region,
            'Preexisting_Condition': data.Preexisting_Condition,
            'COVID_Strain': data.COVID_Strain,
            'Symptoms': data.Symptoms,
            'Severity': data.Severity,
            'ICU_Admission': data.ICU_Admission,
            'Ventilator_Support': data.Ventilator_Support,
            'Recovered': data.Recovered,
            'Reinfection': data.Reinfection,
            'Vaccination_Status': data.Vaccination_Status,
            'Doses_Received': data.Doses_Received,
            'Occupation': data.Occupation,
            'Smoking_Status': data.Smoking_Status,
            'BMI': data.BMI
        }
        data.model_dump()
        
        # Create DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        # Convert prediction back to original labels if label encoder exists
        if label_encoder:
            predicted_label = label_encoder.inverse_transform(prediction)[0]
            class_labels = label_encoder.classes_
        else:
            predicted_label = int(prediction[0])
            class_labels = ["Not Hospitalized", "Hospitalized"]
        
        # Get probability for each class
        probabilities = {
            str(class_labels[i]): float(prediction_proba[0][i]) 
            for i in range(len(class_labels))
        }
        
        return {
            "prediction": predicted_label,
            "prediction_numeric": int(prediction[0]),
            "probabilities": probabilities,
            "confidence": float(max(prediction_proba[0])),
            # "input_received": input_dict
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/predict-batch/")
def predict_batch(data_list: list[CovidPatientFeatures]) -> Dict[str, Any]:
    """Predict for multiple patients at once"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        
        for i, data in enumerate(data_list):
            input_dict = {
                'Age': data.Age,
                'Gender': data.Gender,
                'Region': data.Region,
                'Preexisting_Condition': data.Preexisting_Condition,
                'COVID_Strain': data.COVID_Strain,
                'Symptoms': data.Symptoms,
                'Severity': data.Severity,
                'ICU_Admission': data.ICU_Admission,
                'Ventilator_Support': data.Ventilator_Support,
                'Recovered': data.Recovered,
                'Reinfection': data.Reinfection,
                'Vaccination_Status': data.Vaccination_Status,
                'Doses_Received': data.Doses_Received,
                'Occupation': data.Occupation,
                'Smoking_Status': data.Smoking_Status,
                'BMI': data.BMI
            }
            
            input_df = pd.DataFrame([input_dict])
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)
            
            if label_encoder:
                predicted_label = label_encoder.inverse_transform(prediction)[0]
            else:
                predicted_label = int(prediction[0])
            
            predictions.append({
                "patient_index": i,
                "prediction": predicted_label,
                "prediction_numeric": int(prediction[0]),
                "confidence": float(max(prediction_proba[0]))
            })
        
        return {
            "total_patients": len(data_list),
            "predictions": predictions
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Batch prediction error: {str(e)}"
        )

# Error handlers
@app.exception_handler(404)
def not_found_handler(request, exc):
    return {
        "error": "Endpoint not found",
        "available_endpoints": ["/", "/predict/", "/health/", "/model-info/"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)