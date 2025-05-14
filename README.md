# 🧠 MNIST Digit Classifier

A machine learning web application that classifies handwritten digits using the MNIST dataset. This repository includes both the **model training pipeline** (in Jupyter Notebook) and the [**Flask-based web app**](https://digit-classifier-96wd.onrender.com/).

---

## 📌 Features

- 🧮 Trains a digit classifier with ~98% accuracy on MNIST
- 🌐 Interactive web interface for drawing and predicting digits
- 💾 Model serialization with `joblib`
- ☁️ Deployment-ready on Render using `render.yaml`

---

## 🛠️ Tech Stack

- **Language**: Python
- **Machine Learning**: scikit-learn, NumPy, pandas
- **App Framework**: Flask
- **Frontend**: HTML, CSS, Javascript (static & template folder)
- **Deployment**: Render

---

## 🚀 Getting Started

### 1. Clone the Repository

```py 
git clone https://github.com/Auwal007/MNIST.git
cd MNIST
```
### 2. Set Up Environment

```py
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```
### 3. Run the App

```py
python app.py
```
Open http://localhost:5000 in your browser.

---

**📊 Model Training**

- The training is done in `mnist.ipynb`:

- Loads MNIST from `sklearn.datasets`

- Finally trains on `SVC`

- Achieves `~98% accuracy` on the test set

- Saves the model as `mnist_model.joblib` for use in the web app
  
To retrain:
```py
from sklearn.datasets import load_digits
from joblib import dump

# Your training code here
dump(model, 'mnist_model.joblib')
```

### 📁 Project Structure
``` csharp
├── app.py                 # Flask web server
├── mnist.ipynb            # Model training notebook
├── mnist_model.joblib     # Trained model
├── requirements.txt       # Dependencies
├── render.yaml            # Render deployment file
├── static/                # CSS, JS
└── templates/             # HTML pages


```


---
### ⚠️ Known Issues & Notes
Despite achieving **~98% test accuracy** during training, the model performs poorly when [deployed](https://digit-classifier-96wd.onrender.com/), often failing to accurately classify hand-drawn digits from the web app.

I have implemented the **same preprocessing** steps used during training, ensuring the input is resized, grayscaled, and reshaped exactly like the MNIST dataset. Unfortunately, the issue still persists.

This suggests the problem might be deeper, possibly related to how canvas input is captured or subtle differences in pixel intensity and centering compared to the original dataset or something.

Here is the link to the [deployed web app](https://digit-classifier-96wd.onrender.com/).

### 🙏 Help Wanted
If you're reading this and have experience with ML deployment or working with image inputs from canvas elements, **I'd be very grateful if you could help me figure out what's wrong or suggest improvements**. Any feedback or pull requests are highly appreciated!

### 🌐 Deployment (Render)
To deploy:

- Push this repo to GitHub

- Link the repo to your Render account

- Render will detect render.yaml and set up the service

### 📄 License
MIT License

