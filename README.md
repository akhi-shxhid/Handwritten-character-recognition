# Handwritten-character-recognition
Created a handwritten character recognition system that can recognize various handwritten characters or alphabets.

## Overview
This project implements a Convolutional Neural Network (CNN) for handwritten digit recognition using the MNIST dataset. The system can recognize digits from 0-9 with high accuracy.

## Features
- Data preprocessing and normalization
- CNN model architecture for image classification
- Training and evaluation scripts
- Visualization of predictions
- Model saving and loading capabilities

## Requirements
- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib

## Installation
```bash
pip install tensorflow numpy matplotlib
```

## Project Structure
- `handwritten_digit_recognition.py`: Main script with model training and prediction
- `handwritten_digit_recognition_model.h5`: Saved trained model

## How to Use
### Training the Model
Run the script to train the model:
```bash
python handwritten_digit_recognition.py
```

### Custom Prediction
```python
import tensorflow as tf
from handwritten_digit_recognition import predict_custom_image, preprocess_custom_image

# Load the saved model
model = tf.keras.models.load_model('handwritten_digit_recognition_model.h5')

# Preprocess your custom image (28x28 grayscale)
custom_image = preprocess_custom_image(your_image_array)

# Predict the digit
prediction = predict_custom_image(model, custom_image)
print(f"Predicted Digit: {prediction}")
```

## Dataset
- Uses MNIST Dataset: Handwritten digit database
- 60,000 training images
- 10,000 test images
- 28x28 pixel grayscale images

## Model Architecture
- Convolutional Neural Network (CNN)
- 3 Convolutional layers
- Max pooling layers
- Dropout for regularization
- Softmax output layer

## Extending the Project
- Add more complex preprocessing
- Implement data augmentation
- Extend to recognize full alphabets or words
- Experiment with different model architectures

## Potential Improvements
1. Support for multiple character types
2. Improved preprocessing techniques
3. Transfer learning
4. Real-time prediction interface

## Limitations
- Currently limited to MNIST dataset (digits 0-9)
- Requires clean, preprocessed input images

## Contributing
Contributions are welcome! Please submit pull requests or open issues.