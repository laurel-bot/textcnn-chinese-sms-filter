# TextCNN Chinese SMS Spam Filter

一个基于 **PyTorch + TextCNN** 的中文垃圾短信过滤项目，包含：

- 数据预处理
- 中文分词（`jieba`）
- 词表构建
- 数据集与 DataLoader
- TextCNN 模型训练
- 验证与评估
- 模型保存与加载
- 单条文本预测

## 1. 项目结构

```text
textcnn-chinese-sms-filter/
├── data/
│   └── README.md
├── artifacts/              # 运行后自动生成
├── config.py
├── dataset.py
├── model.py
├── preprocess.py
├── train.py
├── predict.py
├── utils.py
├── requirements.txt
└── README.md
```

## 2. 环境安装

建议使用 Python 3.10+。

```bash
git clone https://github.com/laurel-bot/textcnn-chinese-sms-filter.git
cd textcnn-chinese-sms-filter

python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\\Scripts\\activate    # Windows

pip install -r requirements.txt
```

## 3. 数据格式

请在 `data/` 目录下准备一个 CSV 文件，例如：`data/sms_spam.csv`

要求至少包含以下两列：

- `text`：短信文本
- `label`：标签，垃圾短信为 `1`，正常短信为 `0`

示例：

```csv
text,label
免费领取大奖，点击链接立即领取,1
您好，明天下午三点开会，请准时参加,0
尊敬的用户，您的积分即将过期，回复T退订,1
今晚一起吃饭吗？,0
```

> 如果你的标签是中文（如“垃圾”“正常”）或英文（如 `spam` / `ham`），程序会自动尝试映射。

## 4. 训练模型

默认命令：

```bash
python train.py --data_path data/sms_spam.csv
```

常用参数示例：

```bash
python train.py \
  --data_path data/sms_spam.csv \
  --epochs 10 \
  --batch_size 64 \
  --embed_dim 128 \
  --num_filters 100 \
  --dropout 0.5 \
  --lr 0.001 \
  --max_vocab_size 20000 \
  --max_len 50
```

训练完成后，会在 `artifacts/` 下生成：

- `best_model.pt`：最佳模型参数
- `vocab.json`：词表
- `label_mapping.json`：标签映射
- `train_config.json`：训练配置
- `metrics.json`：评估指标

## 5. 预测单条短信

```bash
python predict.py --text "免费领取1000元补贴，点击链接马上申请"
```

也可以指定模型目录：

```bash
python predict.py \
  --text "您好，您的验证码是 123456" \
  --artifacts_dir artifacts
```

输出示例：

```text
输入文本: 免费领取1000元补贴，点击链接马上申请
预测标签: spam
垃圾短信概率: 0.9821
```

## 6. 代码说明

### `preprocess.py`
负责：
- 文本清洗
- `jieba` 分词
- 词表构建
- 文本转 ID
- padding / truncation

### `dataset.py`
负责：
- 定义 `torch.utils.data.Dataset`
- 返回张量格式的样本

### `model.py`
实现 TextCNN：
- Embedding
- 多卷积核提取 n-gram 特征
- Max Pooling
- 全连接分类

### `train.py`
负责：
- 读取 CSV
- 划分训练集 / 验证集
- 构建 DataLoader
- 模型训练
- 验证集评估
- 保存最佳模型

### `predict.py`
负责：
- 加载模型与词表
- 对单条文本进行预测

## 7. 模型原理简介

TextCNN 适合文本分类任务：

1. 先将中文短信分词并映射到词 ID；
2. 通过 Embedding 层转成稠密向量；
3. 使用多个不同窗口大小的卷积核提取局部语义特征；
4. 每个卷积结果做最大池化；
5. 拼接后输入全连接层完成二分类。

## 8. 可扩展方向

- 替换 `jieba` 为更强的中文分词器
- 加入预训练词向量
- 处理类别不平衡（class weight / focal loss）
- 增加 early stopping
- 输出混淆矩阵与 PR / ROC 指标
- 改造成 FastAPI / Flask 在线推理服务

## 9. 快速运行示例

创建一个最小数据集：

```bash
mkdir -p data
cat > data/sms_spam.csv << 'EOF'
text,label
免费送彩金，点击领取,1
您好，明天记得来拿快递,0
限时优惠，回复1办理贷款,1
晚上一起吃饭,0
EOF
```

然后训练：

```bash
python train.py --data_path data/sms_spam.csv --epochs 5
```

预测：

```bash
python predict.py --text "点击链接领取限时补贴"
```

## 10. License

仅供学习与实验使用。
