import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# importing the inference engine
from src.inference import DefectDetectionEngine

# Global state dictionary to hold the ONNX engine in memory

ml_state= {}

@asynccontextmanager
async def  lifespan(app: FastAPI):
    """Prventing the severe latency"""

    print("---Booting EdgeInspectorAI Server---")

    try:
        # initializing the runtime
        ml_state["engine"] = DefectDetectionEngine()

    except Exception as e:
        print(f"--- [FATAL ERROR] Server failed to start: {e} ---")
        raise e

    yield
    print("--- [SHUTDOWN] Clearing memory and shutting down safely ---")
    ml_state.clear()

# App initialization
app = FastAPI(
    title= "EdgeInspectorAI - Steel Defect Detection API",
    description= "Production API for classifying steel surface defects using ONNX Runtime and MLOps pipelines.",
    version= "1.0.0",
    lifespan= lifespan
)

@app.post("/predict/")
async def predict_defect(file: UploadFile= File(...)):
    """
    Receives an image payload, passes it through the pre-warmed ONNX engine, 
    and returns the mathematical probabilities.
    """
    # reject non-image files
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        # read the image bytes in memory
        image_bytes= await file.read()

        # fetching model
        engine= ml_state["engine"]

        # forward pass
        result= engine.predict(image_bytes)

        return JSONResponse(content= result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
    


