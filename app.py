from flask_cors import CORS
from flask import Flask, request, jsonify
import numpy as np
import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io

# Import your model architecture (ResNet9 from training file)
from model_def import ResNet9

# Load the trained model
modelCrop = pickle.load(open('res/XGBoost.pkl','rb'))

# Load the label encoder used during training
label_encoder = pickle.load(open('res/label_encoder.pkl', 'rb'))

# Load class labels
with open('res/class_labels.pkl', 'rb') as f:
    class_names = pickle.load(f)

app = Flask(__name__)
CORS(app)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet9(3, len(class_names))
model.load_state_dict(torch.load('res/plant-disease-model.pth', map_location=device))
model = model.to(device) 
model.eval()

# Image transform (same as used during training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Prediction function
def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, dim=1)
        predicted_class = class_names[predicted[0].item()]

    return predicted_class

@app.route('/')
def home():
    return jsonify({
        "message": "🌾 Krishi API is running!",
        "endpoints": ["/predict_crop", "/predict_disease"]
    })

@app.route('/predict_crop',methods=['POST'])
def predict_crop():
    try:
        # ✅ Support both JSON and form input
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        # Get input values
        N = float(data.get('N'))
        P = float(data.get('P'))
        K = float(data.get('K'))
        temperature = float(data.get('temperature'))
        humidity = float(data.get('humidity'))
        ph = float(data.get('ph'))
        rainfall = float(data.get('rainfall'))

        # Prepare input for model
        input_query = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Predict encoded label
        predicted_value = modelCrop.predict(input_query)[0]

        # Decode to original crop label
        predicted_crop = label_encoder.inverse_transform([int(predicted_value)])[0]

        return jsonify({'crop': predicted_crop})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if not file.content_type.startswith('image/'):
        return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400
    img_bytes = file.read()
    predicted_class = predict_image(img_bytes)

    return jsonify({'predicted_disease': predicted_class})

if __name__ == '__main__':
    app.run(debug=False, port=5000)
