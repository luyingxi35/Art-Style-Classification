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
        # 遍历文件夹中的所有 JSON 文件
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.json'):  # 只处理 JSON 文件
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'r') as f:
                    file_data = json.load(f)
                    self.data.append(file_data)
        
        # 输入为 5×5 的张量
        self.inputs = [torch.tensor(item['input'], dtype=torch.float32) for item in self.data]
        # 输出为单个 5 维向量
        self.outputs = [torch.tensor(item['label'], dtype=torch.float32) for item in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

# 神经网络模型（输入展平为25维）
class ShallowNN(nn.Module):
    def __init__(self, input_dim=25, hidden_dim=128, output_dim=9):
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

# 训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs=1000):
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
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

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
    model = ShallowNN(input_dim=25, hidden_dim=128)
    
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
    EPOCHS = 500
    HIDDEN_DIM = 128

    # 加载数据
    train_dataset = CustomDataset('data_for_shallow_4')  # 使用文件夹路径
    # test_dataset = CustomDataset('test.json')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 初始化模型
    model = ShallowNN(input_dim=36, hidden_dim=HIDDEN_DIM)
    criterion = nn.MSELoss()  # 适用于回归任务
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练与测试
    train_model(model, train_loader, criterion, optimizer, EPOCHS)
    # test_loss = test_model(model, test_loader, criterion)

    # 保存模型
    torch.save(model.state_dict(), 'shallow_nn_4x4.pth')