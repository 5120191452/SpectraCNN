import os

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from SpectraCNN import MyCNN
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# 定义测试数据集
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # 返回指定索引处的特征和标签
        image=self.features[index].reshape(21, 22)
        # 添加一个颜色通道维度
        image=np.expand_dims(image, axis=0)
        return image, self.labels[index]

# 输入数据路径
data_path = r"Test/test.csv" # 数据
data = np.loadtxt(open(data_path, 'rb'), dtype=np.float32, delimiter=',', skiprows=1,encoding='utf-8-sig')#skiprows=1表示跳过表头

# 划分数据标签
cols = data.shape[1]
X = data[:, :-1]
y = data[:, cols - 1:cols]
y = np.array(y.ravel())  # 数列平铺
y = np.squeeze(y)
print("X", X.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=48)  # 测试集占比0.33

test_dataset = CustomDataset(X, y)

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# 加载模型
model = MyCNN(num_classes=6)


# 加载预训练权重
checkpoint = torch.load("checkpoint/bestmodel.pkl")
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.cuda()
start_epoch = checkpoint['epoch']


model.eval()
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

model.to(device)

# 进行模型预测
true_labels = []
predicted_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device).long()
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(preds.cpu().numpy())

# 计算混淆矩阵
conf_matrix = confusion_matrix(true_labels, predicted_labels,normalize='true')
'''
normalize='true': 这是一个可选参数，用于指定是否以及如何规范化混淆矩阵。normalize参数可以接受以下几个值：
None（默认）: 不对混淆矩阵进行规范化，返回的混淆矩阵中的值是实际的数量。
'true': 将混淆矩阵的每一行（实际类别）规范化为概率，即每行的值除以该行的总和，这样每行的和为1，表示为概率。
'pred': 将混淆矩阵的每一列（预测类别）规范化为概率，即每列的值除以该列的总和，这样每列的和为1，表示为概率。
任何浮点数或整数: 使用该数值将整个混淆矩阵规范化，即每个元素除以这个数值。
'''
# 使用Seaborn绘制混淆矩阵热力图
class_names = ['d4', 'd9', 'd12', 'd122', 'd144', 'j816']
# class_names = ['d4', 'd12', 'd122']
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 18})
plt.xlabel('Predicted Labels', fontsize=18)
plt.ylabel('True Labels', fontsize=18)
# plt.title('Confusion Matrix', fontsize=18)
plt.xticks(rotation=45, ha='right', fontsize=18)
plt.yticks(rotation=45, ha='right', fontsize=18)
plt.show()


class_names = ['d4', 'd9', 'd12', 'd122', 'd144', 'j816']
# class_names = ['d4', 'd12', 'd122']
classification_rep = classification_report(true_labels, predicted_labels, target_names=class_names, digits=4)
# 打印分类报告
print("\nClassification Report:")
print(classification_rep)

#将结果写入文件
output_file_path = 'result/output.txt'
# 添加名称标识
report_name = 'Classification_Report:'
full_classification_rep = f"{report_name}\n\n{classification_rep}"
with open(output_file_path, 'w') as file:
    file.write("Classification Report:")
    file.write(full_classification_rep)