import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def create_confusion_matrix(ground_truth, predicted_values):
    unique_chars = sorted(set("".join(ground_truth + predicted_values)))
    num_classes = len(unique_chars)

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true_plate, pred_plate in zip(ground_truth, predicted_values):
        for true_char, pred_char in zip(true_plate, pred_plate):
            true_idx = unique_chars.index(true_char)
            pred_idx = unique_chars.index(pred_char)
            confusion_matrix[true_idx, pred_idx] += 1

    return confusion_matrix, unique_chars

# Example ground truth and predicted values["23GSX6", "2SXVLX" , "27LHTB", "41GFZ8", "56JTT5", "*7LFJB", "*FHKH0", "72FPRV", "72GSVH", "93PXS9", "76ND**", "99SZG5", "RPNL93", "7D020P", "KHPN44", "XSNB23"]
ground_truth = ["23GSX6", "25XVLX" , "27LHTB", "41GFZ8", "56JTT5", "57LFJB", "63HKHD", "72FPRV", "72GSVH", "93PXS9", "96NDJB", "99SZG5", "RPNL93", "VD020P", "XHPN44", "XSNB23"]
predicted_values = ["23GSX6", "2SXVLX" , "27LHTB", "41GFZ8", "56JTT5", "*7LFJB", "*FHKH0", "72FPRV", "72GSVH", "93PXS9", "76ND**", "99SZG5", "RPNL93", "7D020P", "KHPN44", "XSNB23"]

conf_matrix, unique_chars = create_confusion_matrix(ground_truth, predicted_values)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=unique_chars, yticklabels=unique_chars)
plt.title("Character-wise Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()