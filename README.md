✍️ Handwritten Digit Recognizer

A Deep Learning-based Handwritten Digit Recognition system built using Python, TensorFlow, Keras, and Convolutional Neural Networks (CNN). The project uses the MNIST dataset to train a CNN model that can recognize handwritten digits from 0 to 9.

📌 Project Overview

Handwritten Digit Recognition is a computer vision problem where a machine learning model learns to identify digits from images.

In this project:

The MNIST dataset is used for training and testing.
Images are normalized before training.
A CNN (Convolutional Neural Network) is used to extract image features.
The model is trained to classify digits from 0–9.
The trained model is evaluated on unseen test images.
Predictions are displayed along with the corresponding images.
The trained model is saved as digit_model.keras.
🚀 Features
✅ MNIST handwritten digit dataset
✅ Image preprocessing and normalization
✅ CNN-based deep learning model
✅ Training and validation
✅ Test accuracy evaluation
✅ Digit prediction
✅ Prediction visualization
✅ Trained model saving
🛠️ Technologies Used
Python
TensorFlow
Keras
NumPy
Matplotlib
CNN (Convolutional Neural Network)
📂 Project Structure
Handwritten_Digit_Recognizer/
│
├── digit_recognizer.py
├── requirements.txt
├── README.md
├── digit_model.keras
└── venv/

digit_model.keras will be generated automatically after successful model training.

📊 Dataset

The project uses the MNIST handwritten digit dataset.

The dataset contains:

60,000 training images
10,000 testing images
Image size: 28 × 28 pixels
Classes: 0–9

Each image represents one handwritten digit.

🧠 Model Architecture

The CNN model consists of:

Input Image (28 × 28 × 1)
        ↓
Convolutional Layer (32 filters)
        ↓
Max Pooling
        ↓
Convolutional Layer (64 filters)
        ↓
Max Pooling
        ↓
Flatten
        ↓
Dense Layer (128 neurons)
        ↓
Dropout
        ↓
Output Layer (10 classes)

The final Softmax layer produces probabilities for the 10 possible digits.

⚙️ Installation
1. Clone the Repository
git clone YOUR_GITHUB_REPOSITORY_URL
cd Handwritten_Digit_Recognizer
2. Create Virtual Environment
python -m venv venv
3. Activate Virtual Environment

Windows:

venv\Scripts\activate
4. Install Dependencies
pip install -r requirements.txt
▶️ Run the Project

Run the following command:

python digit_recognizer.py

The program will:

Load the MNIST dataset.
Preprocess the images.
Build the CNN model.
Train the model.
Evaluate the model.
Display test accuracy.
Save the trained model.
Display sample predictions.
📈 Expected Output

After training, the terminal will display information similar to:

TensorFlow Version: 2.x.x

Training images: (60000, 28, 28)
Testing images: (10000, 28, 28)

Epoch 1/5
...
Epoch 5/5
...

Test Accuracy: 98.xx%

Model saved as digit_model.keras

A visualization will also show handwritten digits with their predicted values.

🎯 Example
Input Image → 7
Model Prediction → 7

Input Image → 3
Model Prediction → 3

Input Image → 9
Model Prediction → 9
📌 Learning Outcomes

Through this project, I gained practical experience in:

Convolutional Neural Networks
Computer Vision
Image preprocessing
Deep Learning model training
Classification
TensorFlow and Keras
Model evaluation
Data visualization
🔮 Future Improvements
Add a graphical user interface (GUI).
Allow users to draw digits using a mouse.
Add real-time digit prediction.
Deploy the model as a web application.
Create an API using Flask.
Improve the model using data augmentation.
👨‍💻 Author

Harshal Karad

B.E. – Artificial Intelligence & Data Science

⭐ Project

Handwritten Digit Recognizer using CNN

A deep learning project for recognizing handwritten digits using the MNIST dataset and Convolutional Neural Networks.