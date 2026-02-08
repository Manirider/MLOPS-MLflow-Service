from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union
class PredictRequest(BaseModel):
    image: List[float] = Field(
        description="Flattened image pixel data (784 values for 28x28 MNIST image)"
    )
    @field_validator('image')
    @classmethod
    def validate_image_size(cls, v):
        if len(v) != 784:
            raise ValueError(f"Image must have 784 pixels (28x28), got {len(v)}")
        return v
    @field_validator('image')
    @classmethod
    def validate_pixel_values(cls, v):
        for i, pixel in enumerate(v):
            if not 0 <= pixel <= 255:
                raise ValueError(f"Pixel {i} value {pixel} out of range [0, 255]")
        return v
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "image": [0.0] * 784
                }
            ]
        }
    }
class PredictResponse(BaseModel):
    prediction: int = Field(
        ge=0,
        le=9,
        description="Predicted digit class (0-9)"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Prediction confidence score"
    )
    probabilities: List[float] = Field(
        description="Class probabilities for all digits (0-9)"
    )
    model_name: str = Field(description="Model used for prediction")
    model_version: str = Field(description="Model version used")
    model_stage: str = Field(description="Model stage (e.g., Production)")
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prediction": 7,
                    "confidence": 0.98,
                    "probabilities": [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.98, 0.0, 0.01],
                    "model_name": "MNISTClassifier",
                    "model_version": "3",
                    "model_stage": "Production"
                }
            ]
        }
    }
class BatchPredictRequest(BaseModel):
    images: List[List[float]] = Field(
        description="List of flattened image pixel arrays"
    )
    @field_validator('images')
    @classmethod
    def validate_batch(cls, v):
        if len(v) == 0:
            raise ValueError("At least one image is required")
        if len(v) > 100:
            raise ValueError("Maximum batch size is 100 images")
        for i, img in enumerate(v):
            if len(img) != 784:
                raise ValueError(f"Image {i} must have 784 pixels, got {len(img)}")
        return v
class BatchPredictResponse(BaseModel):
    predictions: List[int] = Field(description="Predicted classes for all images")
    confidences: List[float] = Field(description="Confidence scores for all predictions")
    model_name: str = Field(description="Model used for prediction")
    model_version: str = Field(description="Model version used")
    batch_size: int = Field(description="Number of images processed")
