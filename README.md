# CIFAR-10 图像分类实验

本项目包含自建 CNN、数据增强对比、预训练 ResNet18 微调、CosineAnnealing、早停、测试集评估和 Grad-CAM。

## 环境

课程环境已经安装 PyTorch。Windows PowerShell 中执行：

```powershell
& "D:\asya-ai\hw3env\Scripts\python.exe" -c "import torch; print(torch.cuda.is_available())"
```

CIFAR-10 默认读取 `D:\asya-ai\data`。数据不存在时 `torchvision` 会自动下载到该目录。

## 一键运行

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_all.ps1
```

脚本顺序执行三组实验：`cnn_basic`、`cnn_strong`、`resnet18`。每组训练结束后只加载验证集最优 checkpoint 对官方测试集评估一次，最后为 ResNet18 生成 Grad-CAM。

## 单独运行

```powershell
& "D:\asya-ai\hw3env\Scripts\python.exe" src\train.py `
  --experiment cnn_basic --model cnn --aug basic --epochs 30

& "D:\asya-ai\hw3env\Scripts\python.exe" src\evaluate.py `
  --checkpoint logs\cnn_basic_best.pt --experiment cnn_basic

& "D:\asya-ai\hw3env\Scripts\python.exe" src\gradcam.py `
  --checkpoint logs\cnn_basic_best.pt --experiment cnn_basic
```

`train.py` 支持 `--model`、`--aug`、`--epochs`、`--lr`、`--weight-decay`、`--scheduler`、`--early-stop-patience`、`--batch-size` 和 `--seed`。默认 batch size 为 128，不超过显卡限制。

## 文件和产物

- `src/data.py`：固定 45,000/5,000 划分、基础/强增强、DataLoader。
- `src/model.py`：自建 CNN，以及适配 32×32 输入和 10 类分类头的 ResNet18。
- `src/train.py`：AMP 训练、验证、CosineAnnealing、早停、训练曲线和训练日志。
- `src/evaluate.py`：测试集准确率、混淆矩阵、每类准确率、错误样本拼图。
- `src/gradcam.py`：最后卷积层 Grad-CAM，并在原图上叠加。
- `logs/*_train.json`：训练轨迹、验证集最优指标、训练配置和耗时。
- `logs/*_test.json`：测试集指标、混淆矩阵、错误样本索引。
- `logs/*_gradcam.json`：Grad-CAM 样本元数据。
- `logs/*_best.pt`：验证集最优模型参数。
- `results/*.png`：曲线、混淆矩阵、每类准确率、错误样本和 Grad-CAM。
- `report/report.md`：中文实验报告。

固定随机种子为 42。实验在单张 RTX 4060 Laptop GPU 上使用 batch size 128 和混合精度运行。
