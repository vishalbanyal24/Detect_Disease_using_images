# Medical Image Disease Prediction (Simplified)

Minimal Flask app that loads a pre-trained Vision Transformer (ViT) and predicts one of eight chest conditions from an uploaded image. No training code or dataset setup required.

## Quick Start

FLASK_APP=app.py flask run --host=0.0.0.0 --port=8080
# or
uvicorn app:app --host=0.0.0.0 --port=8080

1. Create a virtual environment (optional)
   ```bash
   python -m venv myvenv
   myvenv\Scripts\activate  # Windows
   # source myvenv/bin/activate  # macOS/Linux
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app
   ```bash
   python app.py
   ```

4. Open your browser at `http://localhost:5000` and upload an image (PNG/JPG or DICOM).

## Notes

- Uploads are stored in `static/uploads` so the UI can preview the image.
- Predictions are based on ImageNet-pretrained ViT with remapped labels; results are for demo/education purposes only.
