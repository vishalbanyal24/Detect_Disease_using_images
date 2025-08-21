import os
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTForImageClassification
import pydicom

# Select compute device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = 'your-secret-key-here'

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions (DICOM CT, plus PNG/JPG images)
ALLOWED_EXTENSIONS = {'dcm', 'png', 'jpg', 'jpeg'}

# Disease classes (based on common chest CT findings)
DISEASE_CLASSES = [
    'Normal',
    'Pneumonia',
    'COVID-19',
    'Tuberculosis',
    'Lung Cancer',
    'Pulmonary Edema',
    'Pleural Effusion',
    'Pneumothorax'
]

# Load the pre-trained model and processor
def load_model():
    try:
        # Using Vision Transformer (ViT) for image classification
        model_name = "google/vit-base-patch16-224"
        processor = ViTImageProcessor.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=len(DISEASE_CLASSES),
            ignore_mismatched_sizes=True
        )
        # Move model to device and set eval mode for inference
        model.to(DEVICE)
        model.eval()
        return model, processor
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# Initialize model globally
model, processor = load_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def _to_float(value, default_value):
    try:
        if isinstance(value, (list, tuple)):
            value = value[0]
        return float(value)
    except Exception:
        return default_value

    


def preprocess_image(image_path):
    """Preprocess DICOM CT images (with windowing) or PNG/JPG images.

    Returns: (image_array, preview_pil, error_message)
    - image_array: numpy array sized for the model (224x224x3)
    - preview_pil: PIL image for UI preview (224x224)
    - error_message: string if validation fails, else None
    """
    try:
        ext = os.path.splitext(image_path)[1].lower()
        if ext == '.dcm':
            ds = pydicom.dcmread(image_path)

            # Enforce CT modality for accuracy
            modality = str(getattr(ds, 'Modality', '')).upper()
            if modality != 'CT':
                raise ValueError('DICOM file is not a CT scan (Modality != CT)')

            # Get raw pixels as Hounsfield-like values
            img = ds.pixel_array.astype(np.float32)

            # Apply rescale slope/intercept if present
            slope = _to_float(getattr(ds, 'RescaleSlope', 1.0), 1.0)
            intercept = _to_float(getattr(ds, 'RescaleIntercept', 0.0), 0.0)
            img = img * slope + intercept

            # Apply windowing (use provided WC/WW, else default lung window)
            wc = _to_float(getattr(ds, 'WindowCenter', None), None)
            ww = _to_float(getattr(ds, 'WindowWidth', None), None)
            if wc is None or ww is None or ww <= 1:
                wc, ww = -600.0, 1500.0  # Lung window default

            lower = wc - (ww / 2.0)
            upper = wc + (ww / 2.0)
            img = np.clip(img, lower, upper)
            img = (img - lower) / (upper - lower)  # normalize 0..1
            img = (img * 255.0).astype(np.uint8)

            # Ensure 3-channel RGB
            if img.ndim == 2:
                img_rgb = np.stack([img] * 3, axis=-1)
            else:
                if img.shape[2] == 1:
                    img_rgb = np.concatenate([img] * 3, axis=-1)
                else:
                    img_rgb = img

            preview_pil = Image.fromarray(img_rgb)
        else:
            # Standard image formats
            preview_pil = Image.open(image_path).convert('RGB')

        preview_pil = preview_pil.resize((224, 224))
        image_array = np.array(preview_pil)
        return image_array, preview_pil, None
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None, None, str(e)

def predict_disease(image_array):
    """Predict disease using the transformer model and return probabilities for all classes."""
    try:
        if model is None or processor is None:
            return None, "Model not loaded", None
        inputs = processor(images=image_array, return_tensors="pt")
        # Move inputs to the same device as the model
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
        predicted_class_id = logits.argmax(-1).item()
        confidence = probabilities[0][predicted_class_id].item()
        predicted_disease = DISEASE_CLASSES[predicted_class_id]
        # Build all class probabilities sorted descending
        sorted_indices = torch.argsort(probabilities[0], descending=True)
        all_predictions = []
        for idx in sorted_indices:
            all_predictions.append({
                'disease': DISEASE_CLASSES[idx.item()],
                'confidence': probabilities[0][idx].item()
            })
        return predicted_disease, confidence, all_predictions
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error='No file selected')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No file selected')
    if file and allowed_file(file.filename):
        try:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Preprocess the image
            image_array, preview_pil, err = preprocess_image(filepath)
            if image_array is None:
                return render_template('index.html', error=err or 'Invalid image. Please upload a valid medical image.')
            # Save preview PNG for UI display
            base_name = os.path.splitext(filename)[0]
            preview_filename = f"{base_name}.png"
            preview_path = os.path.join(app.config['UPLOAD_FOLDER'], preview_filename)
            try:
                preview_pil.save(preview_path)
            except Exception:
                pass
            # Predict disease
            predicted_disease, confidence, all_predictions = predict_disease(image_array)
            if predicted_disease is None:
                return render_template('index.html', error='Error in prediction')
            # Prepare result data
            result = {
                'predicted_disease': predicted_disease,
                'confidence': round(confidence * 100, 2),
                'all_predictions': all_predictions,
                'filename': filename,
                'preview_filename': preview_filename
            }
            return render_template('index.html', result=result)
        except Exception as e:
            return render_template('index.html', error=f'Error processing file: {str(e)}')
    return render_template('index.html', error='Only DICOM CT (.dcm) or PNG/JPG images are accepted')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    # Expect multipart/form-data with a 'file' field
    if 'file' not in request.files:
        return {'error': 'No file provided. Use multipart/form-data with field "file".'}, 400

    file = request.files['file']
    if file.filename == '':
        return {'error': 'No file selected'}, 400
    if not allowed_file(file.filename):
        return {'error': 'Invalid file type'}, 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image_array, _, err = preprocess_image(filepath)
        # Optional: remove after preprocessing
        # os.remove(filepath)

        if image_array is None:
            return {'error': err or 'Please upload a valid DICOM CT (.dcm) or PNG/JPG image'}, 400

        predicted_disease, confidence, all_predictions = predict_disease(image_array)
        if predicted_disease is None:
            return {'error': 'Error in prediction'}, 500

        return {
            'predicted_disease': predicted_disease,
            'confidence': confidence,  # 0-1
            'confidence_percent': round(confidence * 100, 2),  # %
            'predictions': [
                {
                    'disease': p['disease'],
                    'confidence': p['confidence'],
                    'confidence_percent': round(p['confidence'] * 100, 2)
                } for p in all_predictions
            ]
        }, 200
    except Exception as e:
        return {'error': str(e)}, 500

 

@app.route('/health')
def health_check():
    return {'status': 'healthy', 'model_loaded': model is not None, 'device': str(DEVICE)}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 