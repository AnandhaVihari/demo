from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
from models.model_manager import ModelManager

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only. In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager = ModelManager()

class TrainingConfig(BaseModel):
    model_name: str
    training_params: Optional[Dict[str, Any]] = None
    lora_config: Optional[Dict[str, Any]] = None

@app.get("/")
async def root():
    return {"message": "Fine-Tuning Labs API is running"}

@app.get("/api/supported-models")
async def get_supported_models():
    """Get list of supported models"""
    return {"models": list(model_manager.supported_models.keys())}

@app.post("/api/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Handle dataset upload"""
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)
        
        return {"status": "success", "file_path": file_path}
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(e)}
        )

@app.post("/api/start-training")
async def start_training(config: TrainingConfig):
    """Start the fine-tuning process"""
    try:
        output_dir = f"models/{config.model_name}-finetuned"
        os.makedirs(output_dir, exist_ok=True)
        
        result = model_manager.start_training(
            model_name=config.model_name,
            dataset_path=config.dataset_path,
            output_dir=output_dir,
            training_params=config.training_params,
            lora_config=config.lora_config
        )
        
        return result
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(e)}
        )

@app.get("/api/training-config-schema")
async def get_training_config_schema():
    """Get the schema for training configuration"""
    return model_manager.get_training_config_schema()