# 🚗 Deep Learning-Based Image Classification for Vehicles

A deep learning project that classifies vehicle images into categories like cars, bikes, planes, trains, and more using a custom CNN architecture, PyTorch, and Flask for deployment.

## 📂 Dataset

- Source: [Kaggle - Vehicle Classification Dataset](https://www.kaggle.com)
- Description: Labeled images across multiple vehicle classes including:
  - Car
  - Bike
  - Motorcycle
  - Plane
  - Train
  - Ship
  - Auto-Rickshaw

## 🛠️ Project Structure

├── train/                  # Training dataset (organized by class)
├── test/                   # Testing dataset (organized by class)
├── preprocess.py           # Image preprocessing script
├── main.py                 # Model training and evaluation script
├── app.py                  # Flask web application for deployment
├── trained_model.pth       # Trained model weights (handled by Git LFS)
├── static/uploads/         # Upload folder for test images (Flask)
├── templates/              # HTML templates for the web app
└── requirements.txt        # Dependencies


## ⚙️ Preprocessing (`preprocess.py`)

- Resizes all images to 128x128
- Converts images to tensors
- Normalizes pixel values


## 🧠 Model Training (`main.py`)

- Loads and augments training/test datasets:
  - Random flips
  - Rotations
  - Color jittering
- Model architecture: Custom CNN
- Loss function: `CrossEntropyLoss`
- Optimizer: `Adam`
- Final test accuracy: ~61%



## 🌐 Web Deployment (`app.py`)

Built with Flask to classify uploaded images:

1. User uploads an image via web UI
2. Image is preprocessed
3. Trained model predicts vehicle class
4. Output displayed in the browser

Run it locally:

```bash
python app.py
```
Visit: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

## 🚀 Future Work

- Improve accuracy using:
  - Larger dataset
  - Transfer learning (e.g., ResNet, VGG)
- Deploy on cloud platforms like **AWS**, **Azure**, or **Heroku**
- Add image explainability (e.g., Grad-CAM)



## 📦 Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/TanishaAnil/Deep-Learning--Based-Image-Classification-for-Vehicles.git
   cd Deep-Learning--Based-Image-Classification-for-Vehicles
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask app:
   ```bash
   python app.py
   ```



## 🧠 Tech Stack

- Python 3.x
- PyTorch
- Flask
- torchvision
- PIL (Pillow)



## 🏁 Results

- Achieved ~61% classification accuracy
- Successfully deployed model via local web app
