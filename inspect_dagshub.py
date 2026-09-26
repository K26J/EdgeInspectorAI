import os
import mlflow
from mlflow.tracking import MlflowClient
from src.config_parser import cfg

# 1. Authenticate using your flawless Singleton config
os.environ['MLFLOW_TRACKING_USERNAME'] = cfg.mlflow_username
os.environ['MLFLOW_TRACKING_PASSWORD'] = cfg.mlflow_password
mlflow.set_tracking_uri(cfg.mlflow_uri)

client = MlflowClient()
# The Run ID from your original training experiment
run_id = "4cdec6bada614b8b9a15802a584799e6"

print(f"\n[DEBUG] Querying Dagshub for run {run_id}...")
try:
    artifacts = client.list_artifacts(run_id)
    
    if not artifacts:
        print("❌ Dagshub reports NO artifacts at the root of this run!")
    
    for a in artifacts:
        print(f"📂 Root item found: '{a.path}' (Is Directory: {a.is_dir})")
        
        # If it's a directory, look inside it
        if a.is_dir:
            sub_artifacts = client.list_artifacts(run_id, a.path)
            for sa in sub_artifacts:
                print(f"   ↳ Inside '{a.path}': '{sa.path}'")
                
except Exception as e:
    print(f"❌ API Error: {e}")