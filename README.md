# CIFAR-10 Image Classification

本项目基于 CIFAR-10 数据集完成图像分类实验，使用 PyTorch 实现并对比 MLP、CNN 和 ResNet18 三种模型。

## 项目内容

- `train_cifar10.py`：训练单个模型，默认训练 CNN 20 轮。
- `compare_models.py`：依次训练 MLP、CNN、ResNet18，并汇总模型准确率。
- `CIFAR10图像分类实验报告.docx`：实验报告文档。
- `data/`：CIFAR-10 数据集目录，运行时自动下载或读取本地数据，不上传到 GitHub。
- `runs/`：训练结果、曲线图、混淆矩阵等输出目录，不上传到 GitHub。

## 环境依赖

建议使用 Python 3.10 及以上版本。安装依赖：

```bash
pip install -r requirements.txt
```

如需使用 NVIDIA GPU，请安装支持 CUDA 的 PyTorch 版本，可参考 PyTorch 官网命令。

## 运行方式

训练默认 CNN 模型：

```bash
python train_cifar10.py
```

训练指定模型：

```bash
python train_cifar10.py --model mlp
python train_cifar10.py --model cnn
python train_cifar10.py --model resnet18
```

对比三个模型：

```bash
python compare_models.py
```

快速测试代码流程：

```bash
python train_cifar10.py --smoke-test --epochs 1
```

## 默认设置

- 默认模型：CNN
- 默认训练轮数：20
- 默认 batch size：128
- 默认优化器：AdamW
- 默认设备：`auto`，优先使用 CUDA，若不可用则使用 CPU

ResNet18 在默认参数下会自动切换为更适合 CIFAR-10 的 SGD 配置。

## 实验结果

本次实验中，ResNet18 在 20 轮训练设置下取得较好的测试表现，最终测试准确率约为 91.92%。不同模型的具体结果可参考实验报告和 `runs/comparison_summary.json`。
