# Vehicle Image Quantization — Web PBL

A Streamlit web app for Clustering-Driven Image Feature Quantization using your own vehicle photographs. No built-in dataset is used.

## Dataset
Put images in class folders:
```text
dataset/
  Car/
  Bike/
  Bus/
  Truck/
```
Folder names become class labels. Add at least two classes with at least 2 readable images per class; 50–100+ per class is recommended. Use varied viewpoints, lighting and backgrounds. Avoid duplicates.

Supported: JPG, JPEG, PNG, BMP, WEBP.

## Setup (Python 3.9+)
```bash
python -m venv .venv
```
Windows:
```bash
.venv\\Scripts\\activate
```
macOS/Linux:
```bash
source .venv/bin/activate
```
Install and launch:
```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal. Choose **Train & Evaluate** and click **Train / Retrain model**. Then use **Predict Vehicle** to upload an image.

## Method
1. Resize images to 128x128.
2. Make a stratified train/test split.
3. Extract local HOG descriptors from overlapping 32x32 patches.
4. Fit K-Means on training descriptors only.
5. Assign local descriptors to nearest visual-word centers.
6. Represent each image as a normalized visual-word histogram.
7. Train an SVM and compare K=8, 16, 32, 64.
8. Compare with a global-HOG SVM baseline.

## App pages
- Overview: dataset distribution and project workflow.
- Train & Evaluate: train model, compare cluster counts, see accuracy and confusion matrix.
- Predict Vehicle: upload an image and view predicted category and feature histogram.
- Explore Clusters: inspect learned HOG center feature profiles.
- Dataset Guide: instructions for adding images.

## Generated files
- `artifacts/vehicle_model.joblib`
- `outputs/experiment_results.csv`
- `outputs/accuracy_comparison.png`
- `outputs/confusion_matrix.png`
- `outputs/classification_report.txt`
- `outputs/summary.json`

Accuracy is measured on your dataset at training time. Do not report assumed values.
