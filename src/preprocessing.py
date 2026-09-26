
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def unitary_preparation(X_train : np.ndarray, X_test : np.ndarray):
    scaler = MinMaxScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_padded = np.pad(X_train_scaled, pad_width=[(0, 0), (0, 2)])
    X_test_padded = np.pad(X_test_scaled, pad_width=[(0, 0), (0, 2)])

    # L2 normalization for amplitude encoding
    X_train_normalized = (
        X_train_padded /
        np.linalg.norm(X_train_padded, axis=1, keepdims=True)
    )

    X_test_normalized = (
        X_test_padded /
        np.linalg.norm(X_test_padded, axis=1, keepdims=True)
    )
    return X_train_normalized, X_test_normalized
