# 🐾 Cat vs Dog Classifier

A 3-layer neural network built from scratch in PyTorch that classifies cats and dogs, with a live Streamlit web app.

> **Mini Project 4** — Introduction to Machine Learning · Year 3 ITC · I3 AMS S2

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b)

---

## 🌐 Live Demo

[**Try the app live →**](https://your-app-name.streamlit.app)

Upload any photo and the model predicts Cat, Dog, or Other.

---

## 🏗️ Architecture

A 3-layer fully-connected network built from scratch with manual weight matrices.

```
Input  (16384)      ← 128×128 grayscale flattened
   ↓ ReLU
Hidden (128)
   ↓ ReLU
Hidden (64)
   ↓
Output (2)          ← Cat | Dog
```

The web app also detects an "Other" class when the model's confidence is low or split between Cat and Dog.

---

## 📊 Dataset

- **Train:** photos of cats and dogs (the `Train` folder)
- **Validation:** held out during training for hyperparameter tuning
- **Test:** never used during training — measured only for final accuracy

Photos are converted to **128×128 grayscale**, flattened into vectors of 16,384 pixels, and normalized to [0, 1].

> The photos themselves are not included in this repo. To reproduce, organize your own photos as `Train/Cat`, `Train/Dog`, `Test/Cat`, `Test/Dog`, etc.

---

## 🚀 Run locally

```bash
git clone https://github.com/YOUR_USERNAME/cat-dog-classifier.git
cd cat-dog-classifier
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 📁 Project Structure

```
cat-dog-classifier/
│
├── app.py                   # Streamlit web app
├── pro_4.ipynb              # Training notebook
│
├── W1.npy, b1.npy           # Trained weights (input → hidden 1)
├── W2.npy, b2.npy           # (hidden 1 → hidden 2)
├── W3.npy, b3.npy           # (hidden 2 → output)
│
├── requirements.txt         # Python dependencies
├── .gitignore
├── .gitattributes
└── README.md
```

---

## 🧠 How the app handles unknown photos

Because the model only knows two classes, the app uses two confidence checks to detect non-animal photos:

- **Threshold:** the top class must have ≥ 70% confidence
- **Margin:** the gap between Cat and Dog must be ≥ 15%

If either check fails, the app shows the photo as **"Other"** rather than guessing.

---

## 🛠️ Built With

- [PyTorch](https://pytorch.org/) — model and training
- [OpenCV](https://opencv.org/) — image preprocessing
- [Streamlit](https://streamlit.io/) — web app
- [NumPy](https://numpy.org/) — array operations

---

## 📄 License

MIT — feel free to use this as a reference for your own projects.
