
# README
### **Github地址：https://github.com/Kurtor/AI_lab5**
本项目包含 **Lab 5** 实验的相关代码、数据和报告，旨在探索单模态与多模态模型在情感分类任务上的性能对比和融合效果。

---

## 文件结构


```
lab5/
├── data/
│   ├── 1.jpg
│   ├── 1.txt
│   ├── ...
├── data.py                         #之前尝试对数据预处理的文件（未采用）
├── train.txt                       #训练集
├── README.md                       #README
├── requirements.txt                #依赖库
├── 10225501404付震宇lab5实验报告.md   #实验报告
├── img_1.png                       #实验报告使用的图片
├── img_2.png                       #实验报告使用的图片
├── img_3.png                       #实验报告使用的图片
├── test_with_label.txt             #实验生成的预测文件
├── test_without_label.txt          #测试集
├── tem.py                          #消融实验部分代码
└── run.py                          #多模态融合模型实验代码
```

---

## 项目内容

### 1. 数据集
- 数据存储在 `data/` 目录中，包括训练和测试所需的文本文件（`.txt`）和对应的图像文件（`.jpg`）。
- `train.txt`：训练集文件，包含样本标识符和情感标签。
- `test_without_label.txt`：测试集文件，仅包含样本标识符。

### 2. 核心代码
- **`tem.py`**：用于执行单模态（文本或图像）模型的训练与消融实验。
- **`run.py`**：用于多模态融合模型的实验，包括文本和图像特征的联合学习。

### 3. 辅助文件
- **`data.py`**：曾尝试对数据进行预处理，但未采用。
- **`requirements.txt`**：列出了实验环境中所需的 Python 依赖库，可通过以下命令安装：
  ```bash
  pip install -r requirements.txt
  ```
- **`test_with_label.txt`**：实验生成的预测文件，包含测试集样本的预测标签。
- **实验报告**：`10225501404付震宇lab5实验报告.md`，包含实验过程、结果分析和消融实验总结，配有图片展示（`img_1.png`、`img_2.png`、`img_3.png`）。

---

## 使用说明

1. **安装依赖**
   在运行代码前，请确保已安装项目所需的依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. **运行代码**
   - **单模态实验**：运行 `tem.py`，完成单模态模型的训练和验证。
     ```bash
     python tem.py
     ```
   - **多模态融合实验**：运行 `run.py`，执行多模态情感分类任务。
     ```bash
     python run.py
     ```

3. **结果查看**
   - 模型预测结果保存于 `test_with_label.txt`。
   - 实验分析详见 `10225501404付震宇lab5实验报告.md`。

---

## 参考

若需进一步了解实验设计或结果，请参阅实验报告。