import os
import json
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
import torch.nn as nn
from sklearn.metrics import classification_report


# TEST_DATA_PATH = 'test_data_for_shallow'  # 测试数据集路径
TEST_DATA_PATH = '2_test_data_for_shallow'  # 测试数据集路径
IMAGE_PATH = 'testie'
BASELINE_MODEL_PATH = 'fine_tuned_VGG16_180x180.h5'
# SHALLOWNN_PATH = 'shallow_nn_5x5_62.pth'
SHALLOWNN_PATH = 'shallow_nn_2x17.pth'
IS_BASELINE = False # 是否使用基线模型


# 定义
class ShallowNN(nn.Module):
    def __init__(self, input_dim=34, hidden_dim1=256, hidden_dim2=128, hidden_dim3=64, output_dim=17, dropout=0.3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        #self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        #self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.relu3 = nn.ReLU()
        #self.dropout3 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(hidden_dim3, output_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        #x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        #x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        #x = self.dropout3(x)
        x = self.fc4(x)
        return x
    
class ResNetSNN(nn.Module):
    def __init__(self, input_dim=102, hidden_dim1=256, hidden_dim2=128, hidden_dim3=64, output_dim=17, dropout=0.3):
        super().__init__()
        self.flatten = nn.Flatten()
        # 第一层
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        # 第二层
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        # 第三层
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.bn3 = nn.BatchNorm1d(hidden_dim3)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        # 残差连接
        self.shortcut1 = nn.Linear(input_dim, hidden_dim2) if input_dim != hidden_dim2 else nn.Identity()
        self.shortcut2 = nn.Linear(input_dim, hidden_dim3) if input_dim != hidden_dim3 else nn.Identity()
        # 输出层
        self.fc4 = nn.Linear(hidden_dim3, output_dim)

    def forward(self, x):
        x = self.flatten(x)
        out1 = self.fc1(x)
        out1 = self.bn1(out1)
        out1 = self.relu1(out1)
        out1 = self.dropout1(out1)

        out2 = self.fc2(out1)
        out2 = self.bn2(out2)
        # 第一残差
        shortcut1 = self.shortcut1(x)
        out2 = out2 + shortcut1
        out2 = self.relu2(out2)
        out2 = self.dropout2(out2)

        out3 = self.fc3(out2)
        out3 = self.bn3(out3)
        # 第二残差
        shortcut2 = self.shortcut2(x)
        out3 = out3 + shortcut2
        out3 = self.relu3(out3)
        out3 = self.dropout3(out3)

        out = self.fc4(out3)
        return out

class CustomDataset(Dataset):
    def __init__(self, folder_path):
        self.data = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.json'):
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'r') as f:
                    file_data = json.load(f)
                    self.data.append(file_data)
        # 输入为 6×17 的张量
        self.inputs = [item['input'] for item in self.data]
        # 输出为 17维 one-hot
        self.labels = [torch.tensor(item['label'], dtype=torch.float32) for item in self.data]
        # scores为 6×17
        self.scores = [torch.tensor(item['scores'], dtype=torch.float32) for item in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx], self.scores[idx]

def generate_pridiction_for_shallownn(model, scores):
    with torch.no_grad():
        scores = scores.unsqueeze(0)  # (1,6,17)
        pred = model(scores)
    label = np.argmax(np.array(pred))
    return label

def generate_pridiction_for_baseline(model, image_path):
    # 适配你的VGG16输入尺寸
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (180, 180))  # 180需与你模型输入一致
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    label = np.argmax(pred)
    return label

def compute_correction_rate(model, test_loader):
    correct_count = 0
    total_count = 0
    style_total = np.zeros(17)
    style_correct = np.zeros(17)

    all_preds = []
    all_labels = []

    for inputs, labels, scorces in test_loader:
        for idx in range(len(inputs)):
            total_count += 1
            label_one_hot = labels[idx]
            label = np.argmax(np.array(label_one_hot))
            style_total[label] += 1

            if IS_BASELINE:
                input_path = inputs[idx]
                image_path = os.path.join(IMAGE_PATH, input_path)
                print(f"Processing {image_path}({total_count})...")
                predicted_label = generate_pridiction_for_baseline(model, image_path)
            else:
                scores = scorces[idx]
                print(f"Processing {total_count}...")
                predicted_label = generate_pridiction_for_shallownn(model, scores)

            all_preds.append(predicted_label)
            all_labels.append(label)

            if predicted_label == label:
                correct_count += 1
                style_correct[label] += 1

    style_correction_rate = style_correct / style_total

    return correct_count / total_count,  style_correction_rate, style_total, style_correct, all_preds, all_labels

if __name__ == '__main__':
    test_dataset = CustomDataset(TEST_DATA_PATH)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

    if IS_BASELINE:
        model = load_model(BASELINE_MODEL_PATH)
    else:
        model = ResNetSNN(
            input_dim=34,
            hidden_dim1=256,
            hidden_dim2=128,
            hidden_dim3=64,
            output_dim=17,
            dropout=0.3
        )
        model.load_state_dict(torch.load(SHALLOWNN_PATH))
        model.eval()

    accuracy, style_accuracy, style_total, style_correct, all_preds, all_labels = compute_correction_rate(model, test_loader)

    print(f"\nTotal Accuracy: {accuracy * 100:.2f}%\n")

    # 你可以根据你的17类标签名修改labels
    labels = [
        'Minimalism', 'Romanticism', 'Rococo', 'Post_Impressionism', 'Art_Nouveau_Modern',
        'Renaissance', 'Pointillism', 'Realism', 'Ukiyo_e', 'Symbolism', 'Baroque', 'Cubism',
        'Abstract', 'Pop_Art', 'Impressionism', 'Expressionism', 'Color_Field_Painting'
    ]
    for i, name in enumerate(labels):
        if style_total[i] > 0:
            acc = style_accuracy[i] * 100
            print(f"{name} Accuracy: {acc:.2f}% | Total: {style_total[i]} | Correct: {style_correct[i]}")
        else:
            print(f"{name} Accuracy: N/A | Total: 0 | Correct: 0")

    print("\nClassification Report (Precision / Recall / F1-Score):")
    print(classification_report(
        all_labels, all_preds,
        labels=list(range(17)),
        target_names=labels,
        digits=4,
        zero_division=0
    ))

#Baseline Model
#                           precision    recall  f1-score   support

#           Minimalism     1.0000    0.8000    0.8889        20
#          Romanticism     0.7000    0.7000    0.7000        20
#               Rococo     0.8696    1.0000    0.9302        20
#   Post_Impressionism     0.0000    0.0000    0.0000         0
#   Art_Nouveau_Modern     0.0000    0.0000    0.0000         0
#          Renaissance     0.1111    1.0000    0.2000         1
#          Pointillism     0.9524    1.0000    0.9756        20
#              Realism     0.3333    0.2500    0.2857        20
#              Ukiyo_e     0.0000    0.0000    0.0000         0
#            Symbolism     0.5714    0.6000    0.5854        20
#              Baroque     0.8889    0.8000    0.8421        20
#               Cubism     0.8947    0.8500    0.8718        20
#             Abstract     0.0000    0.0000    0.0000         0
#              Pop_Art     0.0000    0.0000    0.0000         0
#        Impressionism     0.7500    0.3000    0.4286        20
#        Expressionism     0.6429    0.4500    0.5294        20
# Color_Field_Painting     0.0000    0.0000    0.0000         0

#            micro avg     0.6766    0.6766    0.6766       201
#            macro avg     0.4538    0.4559    0.4257       201
#         weighted avg     0.7571    0.6766    0.7013       201


# ShallowNN Model
#                       precision    recall  f1-score   support

#           Minimalism     0.9524    1.0000    0.9756        20
#          Romanticism     0.3182    0.3500    0.3333        20
#               Rococo     0.9091    1.0000    0.9524        20
#   Post_Impressionism     0.0000    0.0000    0.0000         0
#   Art_Nouveau_Modern     0.0000    0.0000    0.0000         0
#          Renaissance     0.0000    0.0000    0.0000         1
#          Pointillism     0.7143    0.5000    0.5882        20
#              Realism     0.1111    0.1000    0.1053        20
#              Ukiyo_e     0.0000    0.0000    0.0000         0
#            Symbolism     0.3462    0.4500    0.3913        20
#              Baroque     0.7500    0.4500    0.5625        20
#               Cubism     0.8824    0.7500    0.8108        20
#             Abstract     0.0000    0.0000    0.0000         0
#              Pop_Art     0.0000    0.0000    0.0000         0
#        Impressionism     0.2609    0.3000    0.2791        20
#        Expressionism     0.2692    0.3500    0.3043        20
# Color_Field_Painting     0.0000    0.0000    0.0000         0

#            micro avg     0.5224    0.5224    0.5224       201
#            macro avg     0.3243    0.3088    0.3119       201
#         weighted avg     0.5486    0.5224    0.5276       201

#8071 images 1000 epochs
# Total Accuracy: 63.68%

# Minimalism Accuracy: 95.00% | Total: 20.0 | Correct: 19.0
# Romanticism Accuracy: 45.00% | Total: 20.0 | Correct: 9.0
# Rococo Accuracy: 85.00% | Total: 20.0 | Correct: 17.0
# Post_Impressionism Accuracy: N/A | Total: 0 | Correct: 0
# Art_Nouveau_Modern Accuracy: N/A | Total: 0 | Correct: 0
# Renaissance Accuracy: 0.00% | Total: 1.0 | Correct: 0.0
# Pointillism Accuracy: 75.00% | Total: 20.0 | Correct: 15.0
# Realism Accuracy: 25.00% | Total: 20.0 | Correct: 5.0
# Ukiyo_e Accuracy: N/A | Total: 0 | Correct: 0
# Symbolism Accuracy: 55.00% | Total: 20.0 | Correct: 11.0
# Baroque Accuracy: 75.00% | Total: 20.0 | Correct: 15.0
# Cubism Accuracy: 90.00% | Total: 20.0 | Correct: 18.0
# Abstract Accuracy: N/A | Total: 0 | Correct: 0
# Pop_Art Accuracy: N/A | Total: 0 | Correct: 0
# Impressionism Accuracy: 50.00% | Total: 20.0 | Correct: 10.0
# Expressionism Accuracy: 45.00% | Total: 20.0 | Correct: 9.0
# Color_Field_Painting Accuracy: N/A | Total: 0 | Correct: 0

# Classification Report (Precision / Recall / F1-Score):
#                       precision    recall  f1-score   support

#           Minimalism     1.0000    0.9500    0.9744        20
#          Romanticism     0.5000    0.4500    0.4737        20
#               Rococo     0.8947    0.8500    0.8718        20
#   Post_Impressionism     0.0000    0.0000    0.0000         0
#   Art_Nouveau_Modern     0.0000    0.0000    0.0000         0
#          Renaissance     0.0000    0.0000    0.0000         1
#          Pointillism     0.9375    0.7500    0.8333        20
#              Realism     0.2778    0.2500    0.2632        20
#              Ukiyo_e     0.0000    0.0000    0.0000         0
#            Symbolism     0.4400    0.5500    0.4889        20
#              Baroque     0.6522    0.7500    0.6977        20
#               Cubism     0.8182    0.9000    0.8571        20
#             Abstract     0.0000    0.0000    0.0000         0
#              Pop_Art     0.0000    0.0000    0.0000         0
#        Impressionism     0.4167    0.5000    0.4545        20
#        Expressionism     0.5294    0.4500    0.4865        20
# Color_Field_Painting     0.0000    0.0000    0.0000         0

#            micro avg     0.6368    0.6368    0.6368       201
#            macro avg     0.3804    0.3765    0.3765       201
#         weighted avg     0.6434    0.6368    0.6369       201


# No sparse data snn
# Classification Report (Precision / Recall / F1-Score):
#                       precision    recall  f1-score   support

#           Minimalism     0.0000    0.0000    0.0000         0
#          Romanticism     0.0000    0.0000    0.0000         0
#               Rococo     0.0000    0.0000    0.0000         0
#   Post_Impressionism     0.0000    0.0000    0.0000         0
#   Art_Nouveau_Modern     0.0000    0.0000    0.0000         0
#          Renaissance     0.0000    0.0000    0.0000         0
#          Pointillism     0.0000    0.0000    0.0000         0
#              Realism     0.6897    0.4000    0.5063       100
#              Ukiyo_e     0.0000    0.0000    0.0000         0
#            Symbolism     0.0000    0.0000    0.0000         0
#              Baroque     0.0000    0.0000    0.0000         0
#               Cubism     0.9259    0.8621    0.8929        87
#             Abstract     0.0000    0.0000    0.0000         0
#              Pop_Art     0.0000    0.0000    0.0000         0
#        Impressionism     0.6364    0.4746    0.5437        59
#        Expressionism     0.6176    0.4884    0.5455        86
# Color_Field_Painting     0.0000    0.0000    0.0000         0

#            micro avg     0.5572    0.5572    0.5572       332
#            macro avg     0.1688    0.1309    0.1464       332
#         weighted avg     0.7234    0.5572    0.6244       332

# baseline
# Classification Report (Precision / Recall / F1-Score):
#                       precision    recall  f1-score   support

#           Minimalism     0.0000    0.0000    0.0000         0
#          Romanticism     0.0000    0.0000    0.0000         0
#               Rococo     0.0000    0.0000    0.0000         0
#   Post_Impressionism     0.0000    0.0000    0.0000         0
#   Art_Nouveau_Modern     0.0000    0.0000    0.0000         0
#          Renaissance     0.0000    0.0000    0.0000         0
#          Pointillism     0.0000    0.0000    0.0000         0
#              Realism     0.7391    0.5100    0.6036       100
#              Ukiyo_e     0.0000    0.0000    0.0000         0
#            Symbolism     0.0000    0.0000    0.0000         0
#              Baroque     0.0000    0.0000    0.0000         0
#               Cubism     0.9367    0.8506    0.8916        87
#             Abstract     0.0000    0.0000    0.0000         0
#              Pop_Art     0.0000    0.0000    0.0000         0
#        Impressionism     0.7222    0.4407    0.5474        59
#        Expressionism     0.7347    0.4186    0.5333        86
# Color_Field_Painting     0.0000    0.0000    0.0000         0

#            micro avg     0.5633    0.5633    0.5633       332
#            macro avg     0.1843    0.1306    0.1515       332
#         weighted avg     0.7868    0.5633    0.6509       332
