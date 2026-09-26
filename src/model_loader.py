import os 
import glob
import mlflow
import onnxruntime as ort

class ModelLoader:
    def __init__(self, cfg):
        """Initializes the ModelLoader by securely connecting to the DagsHub MLflow registry."""
        self.model_name= cfg.model_name
        self.model_version = cfg.model_version

        # Set the DagsHub MLflow tracking URI
        mlflow.set_tracking_uri(cfg.mlflow_uri)

        # Explicitly set environment variables
        os.environ['MLFLOW_TRACKING_USERNAME'] = cfg.mlflow_username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = cfg.mlflow_password

    def load_inference_session(self):
        """Loads the ONNX model from the DagsHub MLflow registry"""
        print(f"Connecting to DagsHub... Fetching '{self.model_name}' (Version: {self.model_version})")

        # Construct the Registry URI for the model
        model_uri= f"models:/{self.model_name}/{self.model_version}"

        try:
            local_dir= mlflow.artifacts.download_artifacts(artifact_uri= model_uri)
            onnx_files= glob.glob(os.path.join(local_dir, "**", "*.onnx"), recursive=True)

            if not onnx_files:
                raise FileNotFoundError(f"No ONNX model files found in the downloaded artifacts at {local_dir}")

            onnx_model_path= onnx_files[0]
            print(f"ONNX model found at: {onnx_model_path}. Loading inference session...")

            # Initializing the onnx runtime engine
            session= ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

            input_name= session.get_inputs()[0].name
            print(f"ONNX Runtime Engine initialized successfully. Expected input: '{input_name}'")

            return session

        except Exception as e:
            raise RuntimeError(f"Failed to load model from MLflow registry: {str(e)}")

# Quick test execution
if __name__ == "__main__":
    from src.config_parser import get_config
    
    config = get_config()
    loader = ModelLoader(config)
    inference_session = loader.load_inference_session()