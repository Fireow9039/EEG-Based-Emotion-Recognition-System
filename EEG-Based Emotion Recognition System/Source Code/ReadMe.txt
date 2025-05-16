# EEG-Based Emotion Recognition System

## Project Overview
This project develops a semi-supervised learning model using Variational Autoencoders (VAEs) and Recurrent Neural Networks (RNNs) to recognize emotions from EEG (electroencephalogram) data. The system is designed to handle multi-channel EEG data, employing deep learning techniques to both encode temporal dynamics and decode them to reconstruct the input signals, aiding in feature extraction and classification tasks.

## Features
- Data Normalization: Standardize EEG data to ensure model accuracy and efficiency.
- Advanced Visualization: Implement multiple data visualization strategies to inspect the quality and characteristics of EEG data.
- Semi-supervised Learning Model: Leverage unlabeled data to enhance the model's performance using VAEs integrated with RNNs.
- Model Evaluation: Assess model performance with various metrics, visualizing the comparison between original and reconstructed signals.

## Installation
To set up the project environment:
1. Clone this repository to your local machine.
2. Ensure that you have Python 3.8+ installed.
3. Install all required dependencies with the following command:
	pip install -r requirements.txt

## Usage
To run the main program:
	python main.py

This will execute the data preprocessing, model training, and visualization scripts.

##Dataset
Due to privacy constraints, the EEG dataset utilized in this project cannot be shared publicly. The dataset includes multi-channel EEG recordings annotated with emotional states. Users wishing to test the system should substitute in their dataset following the same structure expected by the scripts.

