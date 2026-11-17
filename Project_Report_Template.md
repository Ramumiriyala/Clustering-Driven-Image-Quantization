# Project Report Template

## Title
Clustering-Driven Image Feature Quantization for Vehicle Pattern Recognition

## Abstract
This project investigates clustering-driven image feature quantization for vehicle recognition using a custom image dataset. Local Histogram of Oriented Gradients (HOG) descriptors are extracted from image patches and grouped using K-Means. Cluster centers form a visual vocabulary. Each image is represented by a normalized histogram of visual words, and an SVM classifier predicts the vehicle category. Different vocabulary sizes are compared with a global-HOG baseline using accuracy, precision, recall, F1-score and a confusion matrix.

## 1. Introduction
Vehicle images vary in viewpoint, scale, illumination and background. Feature quantization represents local descriptors using representative cluster centers. This project studies how vocabulary size affects image representation and downstream recognition.

## 2. Problem Statement
Investigate how representative K-Means cluster centers affect feature quantization and vehicle image classification.

## 3. Objectives
- Prepare a custom vehicle image dataset.
- Extract local HOG descriptors.
- Learn visual words using K-Means.
- Build normalized visual-word histograms.
- Classify vehicle types using SVM.
- Compare cluster counts and a non-quantized baseline.

## 4. Proposed Method
Images are resized and divided into overlapping patches. HOG descriptors represent local gradient patterns. K-Means is fitted only on training descriptors. Each descriptor is assigned to its nearest center. Assignment counts form an image histogram, which is classified using SVM.

## 5. Tools
Python, Streamlit, OpenCV, scikit-image, NumPy, Pandas, scikit-learn, Matplotlib and Plotly.

## 6. Results
Run the project on your own images. Insert measured results from `outputs/summary.json`, `outputs/experiment_results.csv`, `outputs/accuracy_comparison.png`, `outputs/confusion_matrix.png` and `outputs/classification_report.txt`. Do not invent accuracy values.

## 7. Conclusion
Write the conclusion after evaluating measured results. Explain whether quantization improved recognition and how the number of cluster centers affected performance.

## 8. Future Scope
Explore SIFT/ORB descriptors, alternative clustering algorithms, additional classifiers, larger datasets and deep-learning feature representations.

## References
1. N. Dalal and B. Triggs, "Histograms of Oriented Gradients for Human Detection," CVPR, 2005.
2. J. MacQueen, "Some Methods for Classification and Analysis of Multivariate Observations," 1967.
3. C. Cortes and V. Vapnik, "Support-Vector Networks," Machine Learning, 1995.
