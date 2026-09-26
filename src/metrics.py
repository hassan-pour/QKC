from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score
)
import numpy as np
scores_structure = {
        "accuracy": .0,
        "precision_malignant": .0,
        "recall_malignant": .0,
        "f1_malignant": .0,
        "precision_benign": .0,
        "recall_benign": .0,
        "f1_benign": .0,
        "macro_f1": .0,
        "balanced_accuracy": .0
    }

def get_scores(true_y, pred_y):
    scores = scores_structure.copy()
    scores["accuracy"] = accuracy_score(true_y, pred_y)

    scores["precision_malignant"] = precision_score(
        true_y, pred_y, pos_label=0
    )
    scores["recall_malignant"] = recall_score(
        true_y, pred_y, pos_label=0
    )
    scores["f1_malignant"] = f1_score(
        true_y, pred_y, pos_label=0
    )

    scores["precision_benign"] = precision_score(
        true_y, pred_y, pos_label=1
    )
    scores["recall_benign"] = recall_score(
        true_y, pred_y, pos_label=1
    )
    scores["f1_benign"] = f1_score(
        true_y, pred_y, pos_label=1
    )

    scores["macro_f1"] = f1_score(
        true_y, pred_y, average="macro"
    )

    scores["balanced_accuracy"] = balanced_accuracy_score(
        true_y, pred_y
    )

    return scores

def build_kernel(x_train, x_test, kernel_value_function):
    kernel_matrix = np.full((len(x_train), len(x_train)), .0)
    test_matrix = np.full((len(x_test), len(x_train)), .0)
    for i in range(len(x_train) - 1):
        for j in range(i + 1,len(x_train)):
            kernel_matrix[i][j] = kernel_value_function(x_train[i],
                                                       x_train[j])
            kernel_matrix[j][i] = kernel_matrix[i][j]
    for i in range(len(x_train)):
        kernel_matrix[i][i] = 1.0
    for i in range(len(x_test)):
        for j in range(len(x_train)):
            test_matrix[i][j]= kernel_value_function(x_test[i],
                                                    x_train[j])
    return kernel_matrix, test_matrix