import numpy as np

predictions = np.array([
    [0.1, 0.2, 0.7, 0.0, 0.0],
    [0.3, 0.4, 0.3, 0.0, 0.0],
    [0.5, 0.2, 0.3, 0.0, 0.0],
])
h_all = np.zeros(3)
for i in range(3):
    h_all[i] = -np.sum(predictions[i] * np.log2(predictions[i] + 1e-10))  # Avoid log(0)
print(f"h_all: {h_all}")
w_all = 1 / np.exp(h_all)
w_all = w_all / np.sum(w_all)
print(f"w_all: {w_all}")
weighted_preds = predictions * w_all[:, np.newaxis]  # (3, 5) * (3, 1) → (3, 5)
final_prediction = np.sum(weighted_preds, axis=0) / np.sum(w_all)
print(f"final_prediction: {final_prediction}")