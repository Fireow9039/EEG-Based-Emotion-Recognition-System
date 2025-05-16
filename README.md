# EEG-Based-Emotion-Recognition-System


🧠 Project Title: EEG-Based Emotion Recognition System
📌 Goal:
To recognize human emotions using EEG (electroencephalogram) data through a semi-supervised deep learning framework combining:

Variational Autoencoders (VAEs) for feature extraction
Recurrent Neural Networks (RNNs) for modeling temporal EEG signal dependencies
🧩 Key Components:
EEG Data Preprocessing (data_utils.py)
Normalization of EEG data
Likely handles time-series data formatting and splitting
Model Architecture (model.py)
Implements a semi-supervised deep learning model using:
VAE: for unsupervised feature learning
RNN/LSTM: for capturing temporal dependencies in EEG signals
Training & Evaluation (main.py)
Loads and preprocesses EEG data
Trains the VAE-RNN model
Evaluates model performance (possibly using reconstruction loss, accuracy, etc.)
Includes visualization of training and reconstruction results using visualization_utils.py
Data (features_raw.csv)
Contains raw EEG features (likely multi-channel time-series data)
Notebook (NN.ipynb)
May be used for experimentation, debugging, or visualization of model behavior in an interactive way
Environment (requirements.txt)
Uses TensorFlow 2.6.0, scikit-learn, matplotlib, seaborn, etc.
⚙️ How to Use:
Install dependencies:
pip install -r requirements.txt
Run the project:
python main.py
📊 Outcome:
The model attempts to:

Extract latent emotional representations from raw EEG
Classify emotional states (even with limited labeled data)
Visualize signal reconstruction and possibly learned feature spaces
