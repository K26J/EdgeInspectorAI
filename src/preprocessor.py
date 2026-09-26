import io
import numpy as np
from PIL import Image

class ImagePreprocessor:
    def __init__(self, cfg):
        """Initializing the preprocessor using the congifuration object"""
        self.target_size= tuple(cfg.target_size)
        self.mean= np.array(cfg.image_mean, dtype=np.float32)
        self.std= np.array(cfg.image_std, dtype=np.float32)

    def preprocess(self, image_bytes):
        """Transform raw image bytes from HTTP request into an ONNX-ready tensor"""
        # we use io.BytesIO to traet the raw bytes as a file-like object, which PIL can read
        image= Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Resize the image to the target size
        image= image.resize(self.target_size, Image.Resampling.BILINEAR)

        # Convert the image to a numpy array and normalize it
        image_array= np.array(image, dtype=np.float32) / 255.0
        image_array= (image_array - self.mean) / self.std

        # Transpose the array to match the expected input shape for ONNX (C, H, W). Image loads as (H, W, C) by default
        image_array= np.transpose(image_array, (2, 0, 1))

        # The model expects a batch of image so we add dummy batch dimension at axis 0, batch dimesions-> (1, C, H, W)
        input_tensor= np.expand_dims(image_array, axis=0)

        return input_tensor

# Quick check to ensure the mathematical transformation work locally
if __name__=="__main__":
    from config_parser import get_config

    # Generating a fake 500x500 to test the pipeline
    test_image= Image.new('RGB', (500, 500), color='green')
    image_byte_arr= io.BytesIO()
    test_image.save(image_byte_arr, format= 'JPEG')
    fake_payload= image_byte_arr.getvalue()

    # Processing
    config= get_config()
    preprocessor= ImagePreprocessor(config)
    processed_tensor= preprocessor.preprocess(fake_payload)

    print(f"Final Tensor Shape: {processed_tensor.shape}") 
    print(f"Data Type: {processed_tensor.dtype}")


