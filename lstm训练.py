import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import torch
from torch import nn, optim
plt.rcParams['font.sans-serif'] = ["SimHei"]
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import os
import random
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_value = 2020   # 设定随机数种子

np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

torch.manual_seed(seed_value)     # 为CPU设置随机种子
torch.cuda.manual_seed(seed_value)      # 为当前GPU设置随机种子（只用一块GPU）
torch.cuda.manual_seed_all(seed_value)   # 为所有GPU设置随机种子（多块GPU）

torch.backends.cudnn.deterministic = True

# 数据读取和预处理
def pre_data(data_path):
    data = pd.read_csv(data_path)
    data = data.dropna()
    # data = data.drop(0)#第二行数据是单位，不要
    data = data.drop(columns=data.columns[:3])  # 不要前3列
    data["AC"] = data["AC"].astype(float)
    data["CNL"] = data["CNL"].astype(float)
    data["DEN"] = data["DEN"].astype(float)
    data["GR"] = data["GR"].astype(float)
    data["RT"] = data["RT"].astype(float)
    data["RXO"] = data["RXO"].astype(float)
    index_1 = data[data["AC"] < 0].index
    data.drop(index_1, axis=0, inplace=True)
    index_2 = data[data["CNL"] < -5].index
    data.drop(index_2, axis=0, inplace=True)
    index_3 = data[data["DEN"] < 0].index
    data.drop(index_3, axis=0, inplace=True)
    index_4 = data[data["GR"] < 0].index
    data.drop(index_4, axis=0, inplace=True)
    index_5 = data[data["RT"] < 0].index
    data.drop(index_5, axis=0, inplace=True)
    index_6 = data[data["RXO"] < 0].index
    data.drop(index_6, axis=0, inplace=True)

    return data

# #处理超范围数据
def fix_data(data):
    # AC
    ac_lower_bound = 33
    ac_upper_bound = 180
    data.loc[(data["AC"] < ac_lower_bound), "AC"] = ac_lower_bound
    data.loc[(data["AC"] > ac_upper_bound), "AC"] = ac_upper_bound
    
    # CNL
    cnl_lower_bound = -5
    cnl_upper_bound = 50
    data.loc[(data["CNL"] < cnl_lower_bound), "CNL"] = cnl_lower_bound
    data.loc[(data["CNL"] > cnl_upper_bound), "CNL"] = cnl_upper_bound
    
    # DEN
    den_lower_bound = 1.5
    den_upper_bound = 3.01
    data.loc[(data["DEN"] < den_lower_bound), "DEN"] = den_lower_bound
    data.loc[(data["DEN"] > den_upper_bound), "DEN"] = den_upper_bound
    
    # GR
    gr_lower_bound = 3
    data.loc[(data["GR"] < gr_lower_bound), "GR"] = gr_lower_bound
    
    # RT
    rt_lower_bound = 1
    data.loc[(data["RT"] < rt_lower_bound), "RT"] = rt_lower_bound
    
    # RXO
    rxo_lower_bound = 1
    data.loc[(data["RXO"] < rxo_lower_bound), "RXO"] = rxo_lower_bound
    
    return data

# 获取特征和标签
def get_Data(data):
    feature = data[["AC","DEN","GR","RT","RXO"]].values  # 提取特征
    label = data[["CNL"]].values  # 提取标签
    return feature, label

def log_cols(data):
    cols = [2,3,4]
    data[:, cols] = np.where(data[:, cols] <= 0, 1e-9, data[:, cols])
    data[:,cols] = np.log10(data[:,cols].astype(float))
    return data

def normalization(data):
    scaler = MinMaxScaler()
    #scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data,scaler

def create_sequences(X, y, time_steps,step):
    X_seq, y_seq = [], []
    for i in range(0,len(X) - time_steps,step):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

time_steps = 10#滑动窗口大小
step = 1 #滑动窗口步长
input_size = 5 #输入特征数
hidden_size = 64 #隐藏层数
num_layers = 3 #模型层数
dropout_rate = 0.5 
output_size = 1 #输出维度
num_epochs = 50 #训练次数
lr=0.0001 #学习率
batch_size = 200 #批次大小

x = np.empty((1, time_steps, input_size))  # 初始化数据
y = np.empty((1,1))
csv_files = sorted(glob.glob('D:\测井\数据\A_gs地区数据\\*.csv'),key = os.path.getctime)#按生成时间排序
#[sheet_num1:sheet_num2]
i = 0
for file in csv_files[:3]:
    #print(file)
    data = pre_data(file)
    data = fix_data(data)
    i= i+1
    if data.empty:
        continue
    feature, label = get_Data(data)
    feature = log_cols(feature)
    feature,scaler = normalization(feature)
    label,scaler = normalization(label)
    x_, y_ = create_sequences(feature,label, time_steps,step)
    x = np.concatenate((x, x_), axis=0)  # 按行拼接
    y = np.concatenate((y, y_), axis=0)
x = x[1:]
y = y[1:]
print(y.shape)
print(i)

X_train_val, X_test, y_train_val, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=42) # 0.25 x 0.8 = 0.2

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out
    
model = LSTMModel(input_size, hidden_size,num_layers, output_size,dropout_rate)
model = model.to(device)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size)
test_dataloader = DataLoader(test_dataset, batch_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr)

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
    val_loss /= len(val_dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

p=[]
t=[]

#保存模型
model_path = 'D:\\测井\\python代码\\LSTM_test_model.pth'
torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)

model.eval()

with torch.no_grad():
    for inputs, targets in test_dataloader:
        outputs = model(inputs)
        for o in outputs:
            p.append(o.item())
        for ta in targets:
            t.append(ta.item())

#入模井效果
y = np.array(t)
y_pred = np.array(p)
r2 = r2_score(y, y_pred)

mae = mean_absolute_error(y, y_pred)

rmse = mean_squared_error(y, y_pred, squared=False)

test = np.array(y).flatten()
pred = np.array(y_pred).flatten()
corr, p_value = pearsonr(test, pred)
print("Pearson correlation coefficient:", corr)
print("p-value:", p_value)
print("R^2:", r2)
print("MAE:", mae)
print("RMSE:", rmse)
plt.figure(figsize=(10, 6))
plt.plot(y[:100])
plt.plot(y_pred[:100])
plt.legend(('real', 'predict'), fontsize='15')
plt.show()
