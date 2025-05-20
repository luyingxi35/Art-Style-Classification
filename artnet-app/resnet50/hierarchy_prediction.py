# Pipeline for hierarchical patch-based style classification
# 1. Divide the image into 3 layers of patches
# 2. Predict each patch with base model -> 3 arrays of 5d vectors
# 3. Optimize each layer by MRF
# 4. Entropy-based exponential fusion

import os
import json
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import tensorflow as tf
from sklearn.metrics import classification_report
from PIL import Image

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

CNN_INPUT_SZ = 325
TEST_DATA_PATH = 'resnet_test_data_for_shallow' 
IMAGE_PATH = '../testie'
MODEL_PATH = 'models/ResNet50V2_ArtStyleClass.h5'  
MODEL_4_PATH = 'shallow_nn_4x4_25_8.pth'

LABEL_TO_INDEX = {
    'Abstract Art': 0,
    'Cubism': 1,
    'Expressionism': 2,
    'Impressionism': 3,
    'Realism': 4
}

class ShallowNN(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=128, hidden_dim1=64, output_dim=5):
        super().__init__()
        self.flatten = nn.Flatten()  # 将5×5输入展平为25维
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim1)
        # self.fc3 = nn.Linear(hidden_dim1, output_dim)

    def forward(self, x):
        x = self.flatten(x)  # 展平操作：形状从 (batch,5,5) 变为 (batch,25)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.relu(x)
        # x = self.fc3(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, folder_path):
        self.data = []
        # 遍历文件夹中的所有 JSON 文件
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.json'):  # 只处理 JSON 文件
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'r') as f:
                    file_data = json.load(f)
                    self.data.append(file_data)
        
        # 输入为 5×5 的张量
        self.inputs = [item['input'] for item in self.data]
        # 输出为单个 5 维向量
        self.labels = [torch.tensor(item['label'], dtype=torch.float32) for item in self.data]
        self.scores = [torch.tensor(item['scores'], dtype=torch.float32) for item in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx], self.scores[idx]

def patch_dividing_layer_1(pil_image, cnn_input_size=CNN_INPUT_SZ):
    target = 2 * cnn_input_size

    w, h = pil_image.size
    scale = target / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = pil_image.resize((new_w, new_h), Image.BILINEAR)

    # 转为 NumPy，BGR
    resized = np.array(resized_image)
    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    elif resized.shape[2] == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)

    pad_w = target - new_w
    pad_h = target - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=[0, 0, 0])

    p = cnn_input_size
    patches = [
        padded[0:p,       0:p      ],
        padded[0:p,       -p:      ],
        padded[-p:,       0:p      ],
        padded[-p:,       -p:      ]
    ]

    # 转为 PIL Image
    pil_patches = [Image.fromarray(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)) for p in patches]
    return pil_patches

def patch_dividing(image, cnn_input_size=CNN_INPUT_SZ):
    patches_all = []
    patches_layer1 = []
    patches_layer1.append(image)
    patches_layer2 = patch_dividing_layer_1(image, cnn_input_size)

    patches_all.append(patches_layer1)
    patches_all.append(patches_layer2)
    return patches_all

def generate_scores(patches_all, model):
    scores_all = []
    for patches in patches_all:
        scores_tmp = []
        for patch in patches:
            image = patch.convert('RGB').resize((224, 224))
            image_array = np.asarray(image) / 255.0
            image_batch = np.expand_dims(image_array, axis=0)
            pred = model.predict(image_batch)
            scores_tmp.append(pred)
        scores_all.append(scores_tmp)

    return scores_all

def generate_predictions_for_each_lateyr(scores_all, base_model, model_4):
    predictions = np.zeros((2, 5))
    
    with torch.no_grad():
        predictions[0] = np.array(scores_all[0])

        scores_4 = torch.tensor(scores_all[1]).unsqueeze(0).float()
        predictions[1] = model_4(scores_4).squeeze().cpu().numpy()

    return predictions



def compute_p_and_w_per_layer(scores):
    # predicted_idx = np.argmax(scores, axis=1)
    # print(f"Predicted indices: {predicted_idx}") 
    # p_k = []
    # for idx in range(5):
    #     p = np.sum(predicted_idx == idx) / len(predicted_idx)
    #     p_k.append(p)
    #     # print(f"Layer {idx}: p_{idx} = {p}")
    p_k = np.mean(scores, axis=0)
    p_k = np.array(p_k)
    h_k = -np.sum(p_k * np.log2(p_k + 1e-10))  # Avoid log(0)
    w_k = 1 + 1 / np.exp(h_k)
    # print(f"p_k: {p_k}")
    # print(f"Weight: {w_k}")
    
    return p_k, w_k

def compute_p_and_w(scores_all):
    p_all = []
    w_all = []
    for scores in scores_all:
        p_k, w_k = compute_p_and_w_per_layer(scores)
        p_all.append(p_k)
        w_all.append(w_k)
        
    return p_all, w_all

def generate_final_label(predictions, p_all,  w_all):
    # 确保 w_all 是一维的 shape (3,)
    # w_all = np.squeeze(w_all)  # 把 (3,1) -> (3,)
    predictions = np.array(predictions)
    # print(predictions)
    # Normalize the p_all and predictions
    p_all = np.array(p_all)
    p_all_copy = []
    for i in range (len(p_all)):
        p_all_copy.append(p_all[i][0])
    p_all = np.array(p_all_copy)
    # print(p_all)
    for i in range(len(p_all)):
        p_all[i] = p_all[i] / np.sum(p_all[i])
        predictions[i] = predictions[i] / np.sum(predictions[i])
    
    # Ensure predictions are in the range [0, 1]
    for i in range(len(p_all)):
        for j in range(len(p_all[i])):
            if predictions[i][j] < 0:
                predictions[i][j] = 0
            if predictions[i][j] > 1:
                predictions[i][j] = 0.999
            if p_all[i][j] < 0:
                p_all[i][j] = 0.0001
            if p_all[i][j] > 1:
                p_all[i][j] = 0.999
    h_all = np.zeros(2)
    for i in range(2):
        # Choise 1: compute the entropy by the output of shalowNN
        h_all[i] = -np.sum(predictions[i] * np.log2(predictions[i] + 1e-10))  # Avoid log(0)

        # Choise 2: compute the entropy by the output of base model
        # h_all[i] = -np.sum(p_all[i] * np.log2(p_all[i] + 1e-10))  # Avoid log(0)
    w_all = 1 / np.exp(h_all)

    # Choice 1: randomly choose the layer proportional to the weights
    # w_all = w_all / np.sum(w_all)
    # valid_idx = np.argmin(h_all)
    # # valid_idx = np.random.choice(len(w_all), p=w_all)
    # return np.argmax(predictions[valid_idx])

    # Choice 2: weighted average
    # weighted_preds = predictions * w_all[:, np.newaxis]  # (3, 5) * (3, 1) → (3, 5)
    # final_prediction = np.sum(weighted_preds, axis=0) / np.sum(w_all)
    # return np.argmax(final_prediction)

    # Choice 3: simply choose the layer with the lowerest entropy
    # valid_idx = np.argmin(h_all)
    # if h_all[0] < h_all[1]:
    #     valid_idx = 0
    # else:
    #     valid_idx = 1
    # return np.argmax(predictions[valid_idx])

    # Choice 4: weighted average on 1 and 2 layers
    w_all = w_all / (w_all[0] + w_all[1])
    weighted_preds = predictions * w_all[:, np.newaxis]  # (3, 5) * (3, 1) → (3, 5)
    final_prediction = weighted_preds[0] + weighted_preds[1]
    return np.argmax(final_prediction)

def predict_per_image(img_path, model, model_4):
    if not os.path.exists(img_path):
        #print(f"图片不存在: {img_path}")
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = Image.open(img_path)

    # 1. Divide patches
    patches_all = patch_dividing(img)

    # 2. Generate scores
    scores_all = generate_scores(patches_all, model)

    # 3. Generate prdictions
    predictions = generate_predictions_for_each_lateyr(scores_all, model, model_4)

    # 4. Entropy-based exponential fusion
    p_all, w_all = compute_p_and_w(scores_all)

    # 5. Final prediction
    label = generate_final_label(predictions, p_all, w_all)

    return label

def classification_report_to_markdown(y_true, y_pred, target_names):
    report_dict = classification_report(y_true, y_pred, target_names=target_names, digits=4, output_dict=True)
    df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={'index': 'Class'})
    for col in ['precision', 'recall', 'f1-score']:
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    df['support'] = df['support'].astype(int)
    print("\n### Classification Report (Markdown Format):\n")
    print(df.to_markdown(index=False))

def compute_correction_rate(model, model_4, test_loader):
    correct_count = 0
    total_count = 0
    style_total = np.zeros(5)
    style_correct = np.zeros(5)

    all_preds = []
    all_labels = []

    for inputs, labels, scores in test_loader:
        for idx in range(len(inputs)):
            total_count += 1
            label_one_hot = labels[idx]
            label = np.argmax(np.array(label_one_hot))
            style_total[label] += 1
            input_path = inputs[idx]
            image_path = os.path.join(IMAGE_PATH, input_path)
            print(f"Processing {image_path}({total_count}/500)...")
            predicted_label = predict_per_image(image_path, model, model_4)
            if predicted_label == label:
                correct_count += 1
                # print(f"Correct!")
                style_correct[label] += 1
            else:
                # print(f"Wrong.")
            style_correction_rate = style_correct / style_total
            # print(f"Style correction rate: {style_correction_rate}")
            # print(f"Total correction rate: {correct_count / total_count}")
            with open("results_2.txt", "a") as f:
                f.write(f"Style correction rate: {style_correction_rate}\n")
                f.write(f"Total correction rate: {correct_count / total_count}\n")

            all_preds.append(predicted_label)
            all_labels.append(label)
            

    style_correction_rate = style_correct / style_total

    return correct_count / total_count, style_correction_rate, style_total, style_correct, all_preds, all_labels

if __name__ == '__main__':
    base_model = load_model(MODEL_PATH, compile=False)

    model_4 = ShallowNN()
    model_4.load_state_dict(torch.load(MODEL_4_PATH, weights_only=True))
    model_4.eval()

    test_dataset = CustomDataset(TEST_DATA_PATH)  # 使用文件夹路径
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True)

    accuracy, style_accuracy, style_total, style_correct, all_preds, all_labels = compute_correction_rate(base_model, model_4, test_loader)

    print(f"\nTotal Accuracy: {accuracy * 100:.2f}%\n")

    labels = [
    'Abstract Art',
    'Cubism',
    'Expressionism',
    'Impressionism',
    'Realism'
    ]
    for i, name in enumerate(labels):
        print(f"{name} Accuracy: {style_accuracy[i] * 100:.2f}% | Total: {style_total[i]} | Correct: {style_correct[i]}")

    print("\nClassification Report (Precision / Recall / F1-Score):")
    print(classification_report(all_labels, all_preds, target_names=labels, digits=4))

    output_file = "results_2.txt"
    with open(output_file, "w") as f:
        for i, name in enumerate(labels):
            f.write(f"{name} Accuracy: {style_accuracy[i] * 100:.2f}% | Total: {style_total[i]} | Correct: {style_correct[i]}\n")

        f.write("\nClassification Report (Precision / Recall / F1-Score):\n")
        f.write(classification_report(all_labels, all_preds, target_names=labels, digits=4))

    
