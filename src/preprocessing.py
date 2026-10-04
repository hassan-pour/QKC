
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def efficient_su2_preparation(X_train : np.ndarray, X_test : np.ndarray):
    scaler = MinMaxScaler()
    
    X_train_scaled = scaler.fit_transform(X_train) * np.pi
    X_test_scaled = scaler.transform(X_test) * np.pi
    
    X_train_padded = np.pad(X_train_scaled, pad_width=[(0, 0), (0, 2)])
    X_test_padded = np.pad(X_test_scaled, pad_width=[(0, 0), (0, 2)])

    
    return X_train_padded, X_test_padded

def angular_preparation(X_train : np.ndarray, X_test : np.ndarray):
    scaler = MinMaxScaler()
        
    X_train_scaled = scaler.fit_transform(X_train) * np.pi
    X_test_scaled = scaler.transform(X_test) * np.pi
    return X_train_scaled, X_test_scaled