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
from PIL import ImageFile
import predict_utils
from predict_utils import process_image, load_checkpoint, predict


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

CNN_INPUT_SZ = 325
TEST_DATA_PATH = 'test_data_for_shallow' 
IMAGE_PATH = '../testie'
# MODEL_PATH = '../model/artnet'
CKPT_PATH = 'checkpoint5.pth'  # 模型路径  
MODEL_4_PATH = 'shallow_nn_4x4.pth'
MODEL_16_PATH = 'shallow_nn_16x16.pth'

LABEL_TO_INDEX = {
    'Art Nouveau (Modern)': 0,
    'Baroque': 1,
    'Expressionism': 2,
    'Impressionism': 3,
    'Post-Impressionism': 4,
    'Rococo': 5,
    'Romanticism': 6,
    'Surrealism': 7,
    'Symbolism': 8
}

class ShallowNN_4(nn.Module):
    def __init__(self, input_dim=36, hidden_dim=128, output_dim=9):
        super().__init__()
        self.flatten = nn.Flatten()  # 将5×5输入展平为25维
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.flatten(x)  # 展平操作：形状从 (batch,5,5) 变为 (batch,25)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class ShallowNN_16(nn.Module):
    def __init__(self, input_dim=80, hidden_dim1=128, hidden_dim2=64, hidden_dim3=32, output_dim=5):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim3, output_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
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

def patch_dividing_layer_2(image, cnn_input_size=CNN_INPUT_SZ):
    target = cnn_input_size * 4  # 确保图像足够切成4x4块（每块cnn_input_size大小）
    h, w = image.shape[:2]
    scale = target / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    pad_w = target - new_w
    pad_h = target - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    # 平均切分为 4x4 个 patch
    patches = []
    p = cnn_input_size
    for i in range(4):
        for j in range(4):
            patch = padded[i * p:(i + 1) * p, j * p:(j + 1) * p]
            patches.append(patch)

    pil_patches = [Image.fromarray(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)) for p in patches]
    return pil_patches

def patch_dividing(image, cnn_input_size=CNN_INPUT_SZ):
    patches_all = []
    patches_layer1 = []
    patches_layer1.append(image)
    patches_layer2 = patch_dividing_layer_1(image, cnn_input_size)
    # patches_layer3 = patch_dividing_layer_2(image, cnn_input_size)

    # print(f"Layer 1: {len(patches_layer1)} patches")
    # print(f"Layer 2: {len(patches_layer2)} patches")
    # print(f"Layer 3: {len(patches_layer3)} patches")
    patches_all.append(patches_layer1)
    patches_all.append(patches_layer2)
    # patches_all.append(patches_layer3)
    # print(f"Total: {len(patches_all)} layers")
    return patches_all

def generate_scores(patches_all, model):
    scores_all = []
    for patches in patches_all:
        scores_tmp = []
        for patch in patches:
            probs, top_labels = predict(patch, model, 9)
            # 创建一个长度为 9 的列表，初始化为 0（因为有 9 个类别）
            prob_vector = [0.0] * 9

            # 根据 LABEL_TO_INDEX 的顺序填充概率值
            for label, prob in zip(top_labels, probs):
                index = LABEL_TO_INDEX[label]
                prob_vector[index] = prob
            scores_tmp.append(prob_vector)
        scores_all.append(scores_tmp)
    # print(f"Total: {len(scores_all)} layers")
    # print(f"Layer 1: {len(scores_all[0])} scores")
    # print(f"Layer 2: {len(scores_all[1])} scores")
    # print(f"Layer 3: {len(scores_all[2])} scores")
    # print(f"Each layer: {len(scores_all[0][0])} classes")
    return scores_all

# def generate_predictions_for_each_lateyr(scores_all, base_model, model_4, model_16):
def generate_predictions_for_each_lateyr(scores_all, base_model, model_4):
    predictions = np.zeros((2, 9))
    
    with torch.no_grad():
        predictions[0] = np.array(scores_all[0])

        scores_4 = torch.tensor(scores_all[1]).unsqueeze(0).float()
        predictions[1] = model_4(scores_4).squeeze().cpu().numpy()

        # scores_16 = torch.tensor(scores_all[2]).unsqueeze(0).float()
        # predictions[2] = model_16(scores_16).squeeze().cpu().numpy()

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

# def generate_final_label(p_all, w_all):
#     prediction = np.zeros(5)
#     intermidiate = np.zeros(5)
#     sum = 0
#     for i in range(5):
#         for k in range(3):
#             intermidiate[i] += w_all[k] * p_all[k][i]
#         sum += intermidiate[i]
#     for i in range(5):
#         prediction[i] = intermidiate[i] / sum
#     # print(f"Final prediction: {prediction}")
#     prediction_idx = np.argmax(prediction)
#     # print(f"Final predicted index: {prediction_idx}")
#     return prediction_idx

def generate_final_label(predictions, p_all,  w_all):
    # 确保 w_all 是一维的 shape (3,)
    # w_all = np.squeeze(w_all)  # 把 (3,1) -> (3,)
    predictions = np.array(predictions)
    # for i in range(len(predictions)):
    #     for j in range(len(predictions[i])):
    #         if predictions[i][j] < 0:
    #             predictions[i][j] = 0
    #         if predictions[i][j] > 1:
    #             predictions[i][j] = 0.999
    h_all = np.zeros(2)
    for i in range(2):
        # Choise 1: compute the entropy by the output of shalowNN
        # h_all[i] = -np.sum(predictions[i] * np.log2(predictions[i] + 1e-10))  # Avoid log(0)

        # Choise 2: compute the entropy by the output of base model
        h_all[i] = -np.sum(p_all[i] * np.log2(p_all[i] + 1e-10))  # Avoid log(0)
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

def predict_use_only_model_4(img_path, model, model_4):
    if not os.path.exists(img_path):
        #print(f"图片不存在: {img_path}")
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = Image.open(img_path)
    patches_layer1 = patch_dividing_layer_1(img)
    scors_layer1 = []
    for image in patches_layer1:
        patches = patch_dividing_layer_1(image)
        scores_tmp = []
        for patch in patches:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            pred = model.predict(np.expand_dims(patch, axis=0))[0]
            scores_tmp.append(pred.tolist())
        scores_tmp = torch.tensor(scores_tmp).unsqueeze(0).float()
        score = model_4(scores_tmp).squeeze().detach().cpu().numpy()
        scors_layer1.append(score)

    scores_layer1 = torch.tensor(scors_layer1).unsqueeze(0).float()
    predictions = model_4(scores_layer1).squeeze().detach().cpu().numpy()

    return np.argmax(predictions)

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
    style_total = np.zeros(9)
    style_correct = np.zeros(9)

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
                print(f"Correct!")
                style_correct[label] += 1
            else:
                print(f"Wrong.")
            style_correction_rate = style_correct / style_total
            print(f"Style correction rate: {style_correction_rate}")
            print(f"Total correction rate: {correct_count / total_count}")
            with open("results_2.txt", "a") as f:
                f.write(f"Style correction rate: {style_correction_rate}\n")
                f.write(f"Total correction rate: {correct_count / total_count}\n")

            all_preds.append(predicted_label)
            all_labels.append(label)
            

    style_correction_rate = style_correct / style_total

    return correct_count / total_count, style_correction_rate, style_total, style_correct, all_preds, all_labels

if __name__ == '__main__':
    base_model, _, _, _ = load_checkpoint(CKPT_PATH)

    model_4 = ShallowNN_4()
    model_4.load_state_dict(torch.load(MODEL_4_PATH, weights_only=True))
    model_4.eval()

    # model_16 = ShallowNN_16()
    # model_16.load_state_dict(torch.load(MODEL_16_PATH, weights_only=True))
    # model_16.eval()

    test_dataset = CustomDataset(TEST_DATA_PATH)  # 使用文件夹路径
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True)

    # accuracy, style_accuracy, style_total, style_correct, all_preds, all_labels = compute_correction_rate(base_model, model_4, model_16, test_loader)
    accuracy, style_accuracy, style_total, style_correct, all_preds, all_labels = compute_correction_rate(base_model, model_4, test_loader)

    print(f"\nTotal Accuracy: {accuracy * 100:.2f}%\n")

    labels = ['Art Nouveau (Modern)', 'Baroque', 'Expressionism', 'Impressionism', 'Post-Impressionism', 'Rococo', 'Romanticism',
         'Surrealism', 'Symbolism']
    for i, name in enumerate(labels):
        print(f"{name} Accuracy: {style_accuracy[i] * 100:.2f}% | Total: {style_total[i]} | Correct: {style_correct[i]}")

    # print("\nClassification Report (Precision / Recall / F1-Score):")
    # print(classification_report(all_labels, all_preds, target_names=labels, digits=4))

    # output_file = "results_2.txt"
    # with open(output_file, "w") as f:
    #     for i, name in enumerate(labels):
    #         f.write(f"{name} Accuracy: {style_accuracy[i] * 100:.2f}% | Total: {style_total[i]} | Correct: {style_correct[i]}\n")

    #     f.write("\nClassification Report (Precision / Recall / F1-Score):\n")
    #     f.write(classification_report(all_labels, all_preds, target_names=labels, digits=4))

    # 获取实际出现的类别标签
    unique_labels = sorted(set(all_labels + all_preds))

    # 根据实际出现的类别标签生成对应的 target_names
    target_names = [labels[i] for i in unique_labels]

    print("\nClassification Report (Precision / Recall / F1-Score):")
    print(classification_report(all_labels, all_preds, labels=unique_labels, target_names=target_names, digits=4))

    output_file = "results_2.txt"
    with open(output_file, "w") as f:
        for i, name in enumerate(labels):
            f.write(f"{name} Accuracy: {style_accuracy[i] * 100:.2f}% | Total: {style_total[i]} | Correct: {style_correct[i]}\n")

        f.write("\nClassification Report (Precision / Recall / F1-Score):\n")
        f.write(classification_report(all_labels, all_preds, labels=unique_labels, target_names=target_names, digits=4))
