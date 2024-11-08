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

class ImprovedTwoDigitRecognizer(nn.Module):
    def __init__(self):
        super(ImprovedTwoDigitRecognizer, self).__init__()
        # Modified conv layers for wider input (56 pixels instead of 28)
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
        
        # Modified FC layers with three outputs (first digit, second digit, combined number)
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 14, 512),  # Note: 14 instead of 7 due to wider input
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        # Separate output layers for each digit and combined number
        self.digit1_out = nn.Linear(256, 10)
        self.digit2_out = nn.Linear(256, 10)
        self.combined_out = nn.Linear(256, 100)  # 0-99 range

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 64 * 7 * 14)
        x = self.fc_layers(x)
        
        digit1 = self.digit1_out(x)
        digit2 = self.digit2_out(x)
        combined = self.combined_out(x)
        
        return digit1, digit2, combined

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImprovedTwoDigitRecognizer().to(device)

# Load the saved model state
checkpoint = torch.load('best_two_digit_recognizer.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def preprocess_image(image_data):
    # Convert base64 to PIL Image
    image = Image.open(io.BytesIO(base64.b64decode(image_data.split(',')[1])))
    
    # Convert to grayscale
    image = image.convert('L')
    
    # Resize to 28x56 (two digits side by side)
    image = image.resize((56, 28))
    
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
            output1, output2, output_combined = model(tensor)
            
            # Get digit predictions
            pred1 = output1.argmax(dim=1).item()
            pred2 = output2.argmax(dim=1).item()
            pred_combined = output_combined.argmax(dim=1).item()
            
            # Get confidence scores
            prob1 = torch.nn.functional.softmax(output1, dim=1)[0][pred1].item()
            prob2 = torch.nn.functional.softmax(output2, dim=1)[0][pred2].item()
            prob_combined = torch.nn.functional.softmax(output_combined, dim=1)[0][pred_combined].item()
        
        return jsonify({
            'digit1': pred1,
            'digit2': pred2,
            'combined': pred_combined,
            'confidence1': round(prob1 * 100, 2),
            'confidence2': round(prob2 * 100, 2),
            'confidenceCombined': round(prob_combined * 100, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)