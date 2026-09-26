import yaml
import os
from dotenv import load_dotenv

# Initializing the dot_env
load_dotenv()

class Config:
    def __init__(self, config_path= 'config/config.yaml'):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file is not found at {config_path}")

        with open(config_path, 'r') as f:
            self._raw_config= yaml.safe_load(f)

        # Model registery configuration
        self.model_name= self._raw_config['model_registry']['registered_name']
        self.model_version= self._raw_config['model_registry']['version']

        # data_prprocessing configuration
        self.target_size= self._raw_config['data_preprocessing']['target_size']
        self.image_mean= self._raw_config['data_preprocessing']['normalize']['mean']
        self.image_std= self._raw_config['data_preprocessing']['normalize']['std']

        # Enviromental variables configuration
        self.mlflow_uri= os.getenv('MLFLOW_TRACKING_URI')
        self.mlflow_username= os.getenv('MLFLOW_TRACKING_USERNAME')
        self.mlflow_password= os.getenv('MLFLOW_TRACKING_PASSWORD')

        # validation
        self._validate_credentials() 

    def _validate_credentials(self):
        """Ensures that all required mlflow enviromental variables are present"""
        missing_vars= []
        if not self.mlflow_uri:
            missing_vars.append('MLFLOW_TRACKING_URI')
        if not self.mlflow_username:
            missing_vars.append('MLFLOW_TRACKING_USERNAME')
        if not self.mlflow_password:
            missing_vars.append('MLFLOW_TRACKING_PASSWORD')

        if missing_vars:
            raise ValueError(f"Missing required environmental variables: {', '.join(missing_vars)}")

# Wrap the instantiation in a function so it can be called exactly when needed
def get_config(config_path='config/config.yaml'):
    return Config(config_path=config_path)

# Only execute if run directly for testing
if __name__ == "__main__":
    cfg = get_config()
    print(f"Successfully loaded configuration for: {cfg.model_name} v{cfg.model_version}")
