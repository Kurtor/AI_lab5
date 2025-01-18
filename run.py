import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# 如果使用 huggingface 的预训练模型
from transformers import BertTokenizer, BertModel

import torchvision.models as models
import torchvision.transforms as transforms

import chardet

# ---------------------------
# 一、全局配置
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-4
SEED = 42

# 设置随机种子，保证可复现
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

TRAIN_TXT_PATH = 'train.txt'
TEST_TXT_PATH = 'test_without_label.txt'
DATA_DIR = 'data'


# ---------------------------
# 二、自动检测文本文件编码
# ---------------------------
def read_text_file(txt_path):
    """
    使用 chardet 自动检测编码并读取文本文件。
    如果检测不到编码，就默认使用 'utf-8'，并对无法解码的字符用 'replace' 处理。
    """
    with open(txt_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        if encoding is None:
            encoding = 'utf-8'
        text = raw_data.decode(encoding, errors='replace')
    return text


# ---------------------------
# 三、数据集定义
# ---------------------------
class MultiModalDataset(Dataset):
    """
    从给定的 (guid, label) 列表中读取对应的文本、图像，并返回多模态特征。
    若没有 label (例如 'null')，则返回 None 或 -1。
    """

    def __init__(self, data_list, data_dir, tokenizer, max_len=128, transform=None):

        self.data_list = data_list
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform

        # label 映射
        self.label2id = {'negative': 0, 'neutral': 1, 'positive': 2}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        guid, label_str = self.data_list[idx]

        # 1. 读取文本（使用自动检测编码）
        txt_path = os.path.join(self.data_dir, f"{guid}.txt")
        text = read_text_file(txt_path).strip()

        # 2. 读取图像
        img_path = os.path.join(self.data_dir, f"{guid}.jpg")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 3. BERT 分词与编码
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # 4. 处理 label
        if label_str is not None and label_str != 'null':
            label_id = self.label2id[label_str]
        else:
            label_id = -1

        return {
            'guid': guid,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image': image,
            'label': label_id
        }


# ---------------------------
# 四、模型定义
# ---------------------------
class MultiModalSentimentModel(nn.Module):
    """
    将文本 (BERT) 特征 与 图像 (ResNet) 特征 拼接后做情感分类。
    """

    def __init__(self, num_classes=3, text_model_name='bert-base-uncased'):
        super(MultiModalSentimentModel, self).__init__()

        # 1. 文本编码器
        self.bert = BertModel.from_pretrained(text_model_name)
        self.text_hidden_size = self.bert.config.hidden_size

        # 2. 图像编码器（ResNet50）
        self.resnet = models.resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        # 3. 融合层 (将文本特征 + 图像特征 拼接后做一次映射)
        fusion_hidden = 256
        self.fusion = nn.Linear(self.text_hidden_size + num_ftrs, fusion_hidden)

        # 4. 分类层
        self.classifier = nn.Linear(fusion_hidden, num_classes)

    def forward(self, input_ids, attention_mask, images):
        # (1) 文本特征：取 [CLS] (pooled_output)
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_text = text_outputs[1]

        # (2) 图像特征
        img_features = self.resnet(images)

        # (3) 拼接
        fusion_input = torch.cat([pooled_text, img_features], dim=1)
        fusion_output = self.fusion(fusion_input)
        fusion_output = nn.ReLU()(fusion_output)

        # (4) 分类
        logits = self.classifier(fusion_output)

        return logits


# ---------------------------
# 五、训练和验证
# ---------------------------
def train_model(model, train_loader, val_loader, epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.to(DEVICE)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            images = batch['image'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch + 1}] Training Loss: {avg_loss:.4f}")

        val_loss, val_acc = evaluate(model, val_loader)
        print(f"[Epoch {epoch + 1}] Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    return model


def evaluate(model, data_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            images = batch['image'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            logits = model(input_ids, attention_mask, images)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples if total_samples else 0.0
    return avg_loss, accuracy


# ---------------------------
# 六、预测并生成输出结果
# ---------------------------
def predict(model, data_loader, id2label):
    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            images = batch['image'].to(DEVICE)
            guids = batch['guid']

            logits = model(input_ids, attention_mask, images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            for guid, pred_id in zip(guids, preds):
                results.append((guid, id2label[pred_id]))

    return results


# ---------------------------
# 七、主流程
# ---------------------------
def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 1. 读取 train.txt (CSV 格式: guid,tag)
    train_data_list = []
    with open(TRAIN_TXT_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            guid, label = line.split(",")
            train_data_list.append((guid, label))

    # 2. 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 3. 构建训练集 Dataset
    dataset = MultiModalDataset(train_data_list, DATA_DIR, tokenizer, max_len=128, transform=transform)

    # 4. 划分训练集 & 验证集 (8:2)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. 初始化多模态模型
    model = MultiModalSentimentModel(num_classes=3, text_model_name='bert-base-uncased')

    # 6. 训练模型
    model = train_model(model, train_loader, val_loader, epochs=EPOCHS)

    # 7. 读取 test_without_label.txt
    test_data_list = []
    with open(TEST_TXT_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            guid, label = line.split(",")
            test_data_list.append((guid, label))

    test_dataset = MultiModalDataset(test_data_list, DATA_DIR, tokenizer, max_len=128, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 8. 预测
    id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    results = predict(model, test_loader, id2label)

    # 9. 写出预测结果
    #   我们要在输出文件中：guid, 预测情感
    #   并保持 test 文件的顺序
    #   因此先把预测结果存成一个 dict[guid] = predicted_label
    res_dict = {guid: pred_label for guid, pred_label in results}

    output_file = 'test_with_label.txt'
    with open(TEST_TXT_PATH, 'r', encoding='utf-8') as fin, \
            open(output_file, 'w', encoding='utf-8') as fout:

        lines = fin.readlines()
        fout.write("guid,tag\n")

        for i, line in enumerate(lines):
            if i == 0:
                continue
            line = line.strip()
            if not line:
                continue
            guid, _ = line.split(",")
            pred_label = res_dict[guid]
            fout.write(f"{guid},{pred_label}\n")

    print(f"预测结果已写入 {output_file}")


if __name__ == "__main__":
    main()
