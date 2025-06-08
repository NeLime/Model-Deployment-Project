from pydantic import BaseModel, Field
from typing import Optional, Union

class CovidPatientFeatures(BaseModel):
    Age: int = Field(..., description="Patient's age", ge=0, le=120)
    Gender: str = Field(..., description="Patient's gender (e.g., Male, Female)")
    Region: str = Field(..., description="Geographic region")
    Preexisting_Condition: str = Field(..., description="Any preexisting medical conditions")
    COVID_Strain: str = Field(..., description="COVID strain variant")
    Symptoms: str = Field(..., description="COVID symptoms experienced")
    Severity: str = Field(..., description="Severity level (e.g., Mild, Moderate, Severe)")
    ICU_Admission: Union[str, bool] = Field(..., description="ICU admission status")
    Ventilator_Support: Union[str, bool] = Field(..., description="Ventilator support needed")
    Recovered: Union[str, bool] = Field(..., description="Recovery status")
    Reinfection: Union[str, bool] = Field(..., description="Reinfection status")
    Vaccination_Status: str = Field(..., description="Vaccination status")
    Doses_Received: int = Field(..., description="Number of vaccine doses received", ge=0)
    Occupation: str = Field(..., description="Patient's occupation")
    Smoking_Status: str = Field(..., description="Smoking status")
    BMI: float = Field(..., description="Body Mass Index", gt=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "Age": 45,
                "Gender": "Male",
                "Region": "North America",
                "Preexisting_Condition": "Diabetes",
                "COVID_Strain": "Delta",
                "Symptoms": "Fever, Cough",
                "Severity": "Moderate",
                "ICU_Admission": "No",
                "Ventilator_Support": "No",
                "Recovered": "Yes",
                "Reinfection": "No",
                "Vaccination_Status": "Fully Vaccinated",
                "Doses_Received": 2,
                "Occupation": "Healthcare Worker",
                "Smoking_Status": "Non-smoker",
                "BMI": 25.5
            }
        }