import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from PIL import Image
import io
import torchvision.transforms as transforms

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class ImprovedDigitRecognizer(nn.Module):
    def __init__(self):
        super(ImprovedDigitRecognizer, self).__init__()
        # Enhanced Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        
        # Enhanced Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc_layers(x)
        return x

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImprovedDigitRecognizer().to(device)

# Load the saved model state
checkpoint = torch.load('best_digit_recognizer.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def preprocess_image(image_data):
    # Convert base64 to PIL Image
    image = Image.open(io.BytesIO(base64.b64decode(image_data.split(',')[1])))
    
    # Convert to grayscale
    image = image.convert('L')
    
    # Resize to 28x28
    image = image.resize((28, 28))
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    return transform(image).unsqueeze(0)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from the request
        image_data = request.json['image']
        
        # Preprocess the image
        tensor = preprocess_image(image_data).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(tensor)
            prediction = output.argmax(dim=1).item()
            
            # Get confidence scores
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence = probabilities[0][prediction].item()
        
        return jsonify({
            'prediction': prediction,
            'confidence': round(confidence * 100, 2)  # Convert to percentage
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)