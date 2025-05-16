import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath)
    data.drop(columns=data.columns[-1], inplace=True)
    data.fillna(data.mean(), inplace=True)
    return data

def normalize_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data
