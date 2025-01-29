import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import traceback
import warnings
from flask import Flask, request, jsonify, send_from_directory
import os
from flask_cors import CORS

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
HEATMAP_FOLDER = 'heatmaps'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).to(DEVICE).eval()

model = InceptionResnetV1(
    pretrained="vggface2",
    classify=True,
    num_classes=1,
    device=DEVICE
)

checkpoint = torch.load("C:\\DeFakeAlert-master\\DeFakeAlert-master\\resnetinceptionv1_epoch_32.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

def predict(image_path):
    try:
        input_image = Image.open(image_path)

        face = mtcnn(input_image)
        if face is None:
            raise Exception('No face detected')

        face = face.unsqueeze(0)  # Add the batch dimension
        face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)

        prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
        prev_face = prev_face.astype('uint8')

        face = face.to(DEVICE)
        face = face.to(torch.float32)
        face = face / 255.0
        face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()

        target_layers = [model.block8.branch1[-1]]
        use_cuda = torch.cuda.is_available()
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
        targets = [ClassifierOutputTarget(0)]

        grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
        face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)

        # Save heatmap image
        heatmap_filename = 'heatmap.png'
        heatmap_path = os.path.join(HEATMAP_FOLDER, heatmap_filename)
        cv2.imwrite(heatmap_path, cv2.cvtColor(face_with_mask, cv2.COLOR_RGB2BGR))

        with torch.no_grad():
            output = torch.sigmoid(model(face).squeeze(0))
            real_prediction = 1 - output.item()
            fake_prediction = output.item()

            prediction = "real" if output.item() < 0.5 else "fake"

            confidences = {
                'real_percentage': real_prediction * 100,
                'fake_percentage': fake_prediction * 100
            }

        return confidences, heatmap_filename

    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
        return {"error": str(e)}, None

@app.route('/')
def index():
    return send_from_directory('', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, 'uploaded_image.jpg')
        file.save(filepath)
        confidences, heatmap_filename = predict(filepath)
        
        if 'error' in confidences:
            return jsonify({'error': confidences['error']})

        heatmap_url = f'/heatmaps/{heatmap_filename}'
        result = {
            'prediction': 'fake' if confidences['fake_percentage'] > 50 else 'real',
            'real_percentage': confidences['real_percentage'],
            'fake_percentage': confidences['fake_percentage'],
            'heatmap_url': heatmap_url
        }
        return jsonify(result)

@app.route('/heatmaps/<filename>')
def serve_heatmap(filename):
    return send_from_directory(HEATMAP_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
