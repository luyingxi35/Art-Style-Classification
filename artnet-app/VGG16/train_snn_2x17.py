import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os  # 添加 os 模块

# 自定义数据集类（关键修改点：处理5×5输入）
class CustomDataset(Dataset):
    def __init__(self, folder_path):
        self.data = []
        self.inputs = []
        self.outputs = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.json'):
                file_path = os.path.join(folder_path, file_name)
                try:
                    with open(file_path, 'r') as f:
                        file_data = json.load(f)
                    # 检查scores和label格式
                    scores = file_data['scores']
                    label = file_data['label']
                    if (isinstance(scores, list) and len(scores) == 2 and all(isinstance(row, list) and len(row) == 17 for row in scores)
                        and isinstance(label, list) and len(label) == 17):
                        self.data.append(file_data)
                        self.inputs.append(torch.tensor(scores, dtype=torch.float32))
                        self.outputs.append(torch.tensor(label, dtype=torch.float32))
                        
                except Exception as e:
                    print(f"跳过无法解析的文件: {file_path}, 错误: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

# 神经网络模型（输入展平为25维）
class ShallowNN(nn.Module):
    def __init__(self, input_dim=34, hidden_dim1=256, hidden_dim2=128, hidden_dim3=64, output_dim=17, dropout=0.3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(hidden_dim3, output_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x
    
class ResNetSNN(nn.Module):
    def __init__(self, input_dim=34, hidden_dim1=256, hidden_dim2=128, hidden_dim3=64, output_dim=17, dropout=0.3):
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

# 训练函数
def train_model(model, train_loader, criterion, optimizer, scheduler=None, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if scheduler is not None:
            scheduler.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

# 测试函数
def test_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    
    print(f'Test Loss: {total_loss/len(test_loader):.4f}')
    return total_loss/len(test_loader)

# 新增代码：加载模型并进行单样本测试
def load_and_test(model_path, input_sample):
    # 1. 初始化模型（必须与训练时的结构完全一致）
    model = ShallowNN(input_dim=34, hidden_dim=128)
    
    # 2. 加载保存的权重
    try:
        model.load_state_dict(torch.load(model_path))
        print("成功加载模型权重")
    except FileNotFoundError:
        print(f"错误：未找到模型文件 {model_path}")
        return None
    except RuntimeError as e:
        print(f"模型加载失败：{str(e)}")
        return None
    
    # 3. 准备输入数据
    try:
        # 将Python列表转换为PyTorch张量
        input_tensor = torch.tensor(input_sample, dtype=torch.float32)
        # 添加批次维度：形状从 (5,5) 变为 (1,5,5)
        input_tensor = input_tensor.unsqueeze(0)
    except Exception as e:
        print(f"输入数据格式错误：{str(e)}")
        return None
    
    # 4. 进行推理
    model.eval()  # 设置评估模式
    with torch.no_grad():
        output = model(input_tensor)
    
    # 5. 转换输出格式
    return output.squeeze(0).tolist()  # 形状从 (1,5) 变为 (5,)

# # 示例用法
# if __name__ == "__main__":
#     # 示例输入（5个5维向量）
#     test_input = [
#         [0.023028505966067314, 0.3783515691757202, 0.053988635540008545, 0.018318481743335724, 0.5263127684593201],
#         [0.0229677502065897, 0.42988523840904236, 0.13038265705108643, 0.04134174808859825, 0.3754225969314575],
#         [0.03772396594285965, 0.3445439636707306, 0.2882290184497833, 0.29661616683006287, 0.0328868068754673],
#         [0.13365407288074493, 0.6305673718452454, 0.07100767642259598, 0.054884351789951324, 0.10988659411668777],
#         [0.03814079239964485, 0.7111823558807373, 0.09215845912694931, 0.08710551261901855, 0.07141280174255371]
#     ]
    
#     # 执行测试
#     prediction = load_and_test('shallow_nn_5x5.pth', test_input)
    
#     if prediction is not None:
#         print("\n测试结果：")
#         print("输入形状：5×5")
#         print("输出预测：", [round(x, 4) for x in prediction])  # 保留四位小数

# 主程序
if __name__ == "__main__":
    # 超参数
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 1000
    HIDDEN_DIM = 128

    # 加载数据
    train_dataset = CustomDataset('2_train_data_for_shallow')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型
    model = ResNetSNN(
        input_dim=34,
        hidden_dim1=256,
        hidden_dim2=128,
        hidden_dim3=64,
        output_dim=17,
        dropout=0.3
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 学习率调度器，每100轮将lr乘以0.5
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

    # 训练
    train_model(model, train_loader, criterion, optimizer, scheduler, EPOCHS)

    # 保存模型
    torch.save(model.state_dict(), 'shallow_nn_2x17.pth')