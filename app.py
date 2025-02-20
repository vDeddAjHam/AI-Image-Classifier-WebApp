from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)
model.eval()

# Define image transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load ImageNet class labels
with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file uploaded", 400
        file = request.files['image']
        if file.filename == '':
            return "No selected file", 400
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Process image
        image = Image.open(filepath).convert('RGB')
        img_t = transform(image)
        batch_t = torch.unsqueeze(img_t, 0)
        with torch.no_grad():
            out = model(batch_t)
        _, index = torch.max(out, 1)
        predicted_label = labels[index[0]]
        return render_template('result.html', label=predicted_label, filename=filename)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename))

if __name__ == "__main__":
    app.run(debug=True)
