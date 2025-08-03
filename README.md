
## ASL Alphabet Recognition with Deep Learning & Transfer Learning

```text
This project focuses on American Sign Language (ASL) alphabet recognition using both traditional machine learning (ML) and deep learning (DL) techniques. It includes the implementation of custom CNN architectures and transfer learning using VGG16, as well as a live webcam-based ASL detection interface.

📌 Features
Classification of 29 ASL alphabet signs (A–Z, space, del, nothing)

Models implemented:

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Custom CNN (with and without data augmentation)

VGG16-based Transfer Learning (frozen & fine-tuned)

GUI-based real-time webcam detection using Tkinter

📁 Directory Structure
.
├── data/
│   ├── asl_alphabet_train/       # Training dataset (29 ASL classes)
│   └── asl_alphabet_test/        # External test dataset
├── model/                        # Saved trained model files (.keras)
├── prediction/                   # Webcam-based real-time ASL prediction GUI
├── SVM_KNN.ipynb                 # Traditional ML models: KNN and SVM
├── CNN.ipynb                     # CNN training (original data)
├── CNN_1.ipynb                   # CNN training (with augmented data)
├── TransferLearning.ipynb        # VGG16 (frozen layers)
├── TransferLearning_1.ipynb      # VGG16 (fine-tuned model)
├── data_processing.py            # Preprocessing and augmentation script
└── README.md

⚙️ Installation

git clone https://github.com/TanzimaSultana/ASL.git
cd ASL

For webcam GUI, ensure:

You have a working webcam & Tkinter and OpenCV are installed

Run Webcam GUI

cd prediction
run real_time_prediciton_1.ipynb

License

This project is for academic and educational purposes. Modify and use freely with attribution.

