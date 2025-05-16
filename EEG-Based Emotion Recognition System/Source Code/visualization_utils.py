import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft
import numpy as np
import pandas as pd

def visualize_data(data, title="Normalized EEG Data"):
    plt.figure(figsize=(15, 7))
    plt.plot(data[:1000])
    plt.title(title)
    plt.xlabel('Sample index')
    plt.ylabel('Normalized amplitude')
    plt.show()

def visualize_histograms(data):
    plt.figure(figsize=(12, 6))
    plt.hist(data, bins=30, alpha=0.75)
    plt.title('Distribution of Features')
    plt.xlabel('Feature Values')
    plt.ylabel('Frequency')
    plt.show()

def visualize_boxplot(data):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=pd.DataFrame(data))
    plt.title('Feature Outliers')
    plt.show()

def visualize_correlation(data):
    correlation_matrix = pd.DataFrame(data).corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def visualize_spectral_density(data, sampling_rate=256):
    yf = fft(data)
    xf = np.linspace(0.0, sampling_rate/2, len(yf)//2)
    plt.figure(figsize=(12, 6))
    plt.plot(xf, 2.0/len(yf) * np.abs(yf[0:len(yf)//2]))
    plt.title('Spectral Density')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.show()

def visualize_multi_channel_time_series(data, num_samples=1000):
    plt.figure(figsize=(15, 10))
    num_channels = data.shape[1]
    for i in range(num_channels):
        plt.subplot(num_channels, 1, i + 1)
        plt.plot(data.iloc[:num_samples, i])
        plt.title('Channel ' + str(i + 1))
    plt.tight_layout()
    plt.show()

def visualize_reconstructions(original, reconstructed, samples=1000):
    plt.figure(figsize=(12, 6))
    time = range(samples)
    plt.plot(time, original[:samples], label='Original')
    plt.plot(time, reconstructed[:samples], label='Reconstructed', alpha=0.7)
    plt.title('Comparison of Original and Reconstructed Signals')
    plt.xlabel('Time')
    plt.ylabel('Signal Amplitude')
    plt.legend()
    plt.show()
