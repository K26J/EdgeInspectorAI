import numpy as np
from src.config_parser import get_config
from src.preprocessor import ImagePreprocessor
from src.model_loader import ModelLoader

class DefectDetectionEngine:
    def __init__(self):

        """Initialize the entire inference pipeline for once during server startup"""
        print("Initializing the Defect Detection Engine...")

        # Configuration
        self.cfg = get_config()

        # Initializing the preprocessor
        self.preprocessor = ImagePreprocessor(self.cfg)

        # Initializing the model loader
        self.model_loader= ModelLoader(self.cfg)

        # Loading onnx runtime
        self.session= self.model_loader.load_inference_session()

        # Extract ONNX metadata 
        self.input_name= self.session.get_inputs()[0].name
        self.output_name= self.session.get_outputs()[0].name

        # class mapping
        self.class_names= ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

    def _softmax(self, x:np.ndarray):
        """
        Mathematically converts raw network logits into probabilities.
        Subtracting np.max(x) provides numerical stability against exploding gradients.
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def predict(self, image_bytes: bytes):
        """
        Takes in raw image bytes, preprocesses it, and returns the predicted class and confidence score.
        """
        # Preprocess the image
        input_tensor= self.preprocessor.preprocess(image_bytes)

        # forward pass
        raw_output= self.session.run([self.output_name], {self.input_name: input_tensor})[0]

        # Convert logits to probabilities
        probabilities= self._softmax(raw_output)[0]

        # Business Logic Translation (Probabilities -> JSON Dictionary)
        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])
        
        return {
            "prediction": self.class_names[predicted_index],
            "confidence": round(confidence, 4),
            "class_probabilities": {
                name: float(round(prob, 4)) for name, prob in zip(self.class_names, probabilities)
            }
        }

# Quick testing execution
if __name__ == "__main__":
    import io
    from PIL import Image
    
    # 1. Initialize the engine (Downloads model, parses config)
    engine = DefectDetectionEngine()
    
    # 2. Create a dummy image payload to test the full pipeline
    test_image = Image.new('RGB', (500, 500), color='gray')
    img_byte_arr = io.BytesIO()
    test_image.save(img_byte_arr, format='JPEG')
    fake_payload = img_byte_arr.getvalue()
    
    # 3. Execute a prediction
    result = engine.predict(fake_payload)
    
    print("\n--- Inference Result ---")
    print(result)