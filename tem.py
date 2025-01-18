import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import BertTokenizer, BertModel
import torchvision.models as models
import torchvision.transforms as transforms

import chardet

# ---------------------------
# 全局配置
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 1e-4
SEED = 42

# 设置随机种子，保证可复现
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

TRAIN_TXT_PATH = 'train.txt'
DATA_DIR = 'data'


# ---------------------------
# 辅助函数：动态检测文件编码
# ---------------------------
def read_text_file(txt_path):
    try:
        with open(txt_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding'] if result['encoding'] else 'utf-8'
            return raw_data.decode(encoding, errors='replace').strip()
    except Exception as e:
        print(f"Error reading file {txt_path}: {e}")
        return ""


# ---------------------------
# 数据集定义
# ---------------------------
class MultiModalDataset(Dataset):
    def __init__(self, data_list, data_dir, tokenizer, transform=None, max_len=128):
        self.data_list = data_list
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len
        self.label2id = {'negative': 0, 'neutral': 1, 'positive': 2}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        guid, label_str = self.data_list[idx]

        txt_path = os.path.join(self.data_dir, f"{guid}.txt")
        text = read_text_file(txt_path)

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

        img_path = os.path.join(self.data_dir, f"{guid}.jpg")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label_id = self.label2id[label_str] if label_str != 'null' else -1
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'image': image, 'label': label_id}


# ---------------------------
# 文本模型
# ---------------------------
class TextOnlySentimentModel(nn.Module):
    def __init__(self, num_classes=3, text_model_name='bert-base-uncased'):
        super(TextOnlySentimentModel, self).__init__()
        self.bert = BertModel.from_pretrained(text_model_name)
        self.text_hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(self.text_hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_text = text_outputs[1]
        logits = self.classifier(pooled_text)
        return logits


# ---------------------------
# 图像模型
# ---------------------------
class ImageOnlySentimentModel(nn.Module):
    def __init__(self, num_classes=3):
        super(ImageOnlySentimentModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, images):
        img_features = self.resnet(images)
        logits = self.classifier(img_features)
        return logits


# ---------------------------
# 通用训练函数
# ---------------------------
def train_single_modal_model(model, train_loader, val_loader, use_text, epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.to(DEVICE)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()
            if use_text:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                logits = model(input_ids, attention_mask)
            else:
                images = batch['image'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_loss, val_acc = evaluate_single_modal_model(model, val_loader, use_text)
        print(
            f"[Epoch {epoch + 1}] Training Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    return model


# ---------------------------
# 通用验证函数
# ---------------------------
def evaluate_single_modal_model(model, data_loader, use_text):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            if use_text:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                logits = model(input_ids, attention_mask)
            else:
                images = batch['image'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                logits = model(images)

            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples if total_samples else 0.0
    return avg_loss, accuracy


# ---------------------------
# 主流程
# ---------------------------
def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_data_list = []
    with open(TRAIN_TXT_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
        for line in lines:
            guid, label = line.strip().split(",")
            train_data_list.append((guid, label))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = MultiModalDataset(train_data_list, DATA_DIR, tokenizer, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Training Text-Only Model...")
    text_model = TextOnlySentimentModel(num_classes=3)
    train_single_modal_model(text_model, train_loader, val_loader, use_text=True)

    print("Training Image-Only Model...")
    image_model = ImageOnlySentimentModel(num_classes=3)
    train_single_modal_model(image_model, train_loader, val_loader, use_text=False)


if __name__ == "__main__":
    main()
