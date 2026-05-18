import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def parse_args():
    # 命令行参数用于控制模型类型、训练轮数、学习率和数据增强等实验变量。
    parser = argparse.ArgumentParser(description="Train MLP or CNN on CIFAR-10.")
    parser.add_argument("--model", choices=["mlp", "cnn", "resnet18"], default="cnn")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--strong-augment", action="store_true", help="Enable AutoAugment and RandomErasing.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset-size", type=int, default=0, help="Use a small subset for quick experiments.")
    parser.add_argument("--no-augment", action="store_true", help="Disable random crop and horizontal flip.")
    parser.add_argument("--smoke-test", action="store_true", help="Use FakeData instead of downloading CIFAR-10.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def set_seed(seed):
    # 固定随机种子，减少每次运行之间的随机波动，便于比较不同模型。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class MLP(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        # MLP 将 32x32x3 图像展平成一维向量，再通过全连接层完成分类。
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        # CNN 保留图像的空间结构，通过卷积层逐级提取局部纹理和形状特征。
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout / 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout / 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_resnet18():
    # 原始 ResNet18 面向较大图像；这里改小首层卷积并去掉最大池化，更适合 CIFAR-10 的 32x32 小图。
    model = models.resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def build_model(model_name, dropout):
    # 根据参数选择不同模型，便于在同一套训练流程中比较 MLP、CNN 和 ResNet18。
    if model_name == "mlp":
        return MLP(dropout)
    if model_name == "resnet18":
        return build_resnet18()
    return SimpleCNN(dropout)


def build_transforms(train, augment, strong_augment):
    steps = []
    if train and augment:
        # 训练集使用随机裁剪和翻转增强泛化能力；测试集不做随机增强，保证评估稳定。
        steps.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
        if strong_augment:
            # AutoAugment 和 RandomErasing 是更强的数据增强，通常能提升泛化，但训练会稍慢。
            steps.append(transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10))
    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    if train and augment and strong_augment:
        steps.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"))
    return transforms.Compose(steps)


def build_datasets(args):
    if args.smoke_test:
        # smoke-test 使用假数据快速检查代码流程，不依赖真实 CIFAR-10 文件。
        train_transform = build_transforms(train=True, augment=not args.no_augment, strong_augment=args.strong_augment)
        test_transform = build_transforms(train=False, augment=False, strong_augment=False)
        train_set = datasets.FakeData(
            size=max(args.subset_size or 256, 64),
            image_size=(3, 32, 32),
            num_classes=10,
            transform=train_transform,
        )
        test_set = datasets.FakeData(
            size=128,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=test_transform,
        )
        return train_set, test_set

    train_set = datasets.CIFAR10(
        # torchvision 会自动读取 data/cifar-10-batches-py 中的二进制批文件。
        root=args.data_dir,
        train=True,
        download=True,
        transform=build_transforms(train=True, augment=not args.no_augment, strong_augment=args.strong_augment),
    )
    test_set = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        transform=build_transforms(train=False, augment=False, strong_augment=False),
    )
    return train_set, test_set


def resolve_training_hparams(args):
    # ResNet18 在 20 轮训练时用 SGD 往往比默认 AdamW 更容易获得较高准确率。
    lr = args.lr
    weight_decay = args.weight_decay
    optimizer_name = args.optimizer

    if args.model == "resnet18" and args.lr == 1e-3 and args.optimizer == "adamw" and args.weight_decay == 1e-4:
        # Better 20-epoch default for ResNet18 on CIFAR-10.
        lr = 0.1
        weight_decay = 5e-4
        optimizer_name = "sgd"

    return optimizer_name, lr, weight_decay


def maybe_subset(dataset, size, seed):
    # 可选抽样训练集，用于在算力有限时快速完成实验对比。
    if size <= 0 or size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


def train_one_epoch(model, loader, criterion, optimizer, device):
    # 单轮训练：前向传播计算损失，反向传播更新参数，并统计训练准确率。
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    # 评估阶段关闭梯度计算，节省显存/内存，并收集预测结果用于分类报告和混淆矩阵。
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "predictions": all_preds,
        "labels": all_labels,
    }


def plot_history(history, run_dir):
    # 保存训练曲线，报告中可用来分析 loss 和 accuracy 的变化趋势。
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["test_loss"], label="test")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="train")
    axes[1].plot(epochs, history["test_accuracy"], label="test")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(run_dir / "loss_accuracy_curves.png", dpi=180)
    plt.close(fig)


def plot_confusion_matrix(labels, preds, run_dir):
    # 混淆矩阵用于观察哪些类别容易被模型混淆。
    matrix = confusion_matrix(labels, preds, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CIFAR10_CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    threshold = matrix.max() / 2 if matrix.max() else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            ax.text(col, row, matrix[row, col], ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(run_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def save_results(args, history, final_eval, best_accuracy, run_dir):
    # 将关键指标保存为 JSON，方便后续写实验报告或做模型对比。
    report = classification_report(
        final_eval["labels"],
        final_eval["predictions"],
        labels=list(range(10)),
        target_names=CIFAR10_CLASSES,
        zero_division=0,
        output_dict=True,
    )
    result = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "best_test_accuracy": best_accuracy,
        "final_test_loss": final_eval["loss"],
        "final_test_accuracy": final_eval["accuracy"],
        "classification_report": report,
        "history": history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    set_seed(args.seed)

    # auto 模式会优先使用 CUDA GPU；没有可用 GPU 时自动回退到 CPU。
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"{timestamp}_{args.model}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_set, test_set = build_datasets(args)
    train_set = maybe_subset(train_set, args.subset_size, args.seed)

    # DataLoader 负责按 batch 读取数据；pin_memory 在 CUDA 训练时可加速数据传输。
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.model, args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer_name, lr, weight_decay = resolve_training_hparams(args)
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=args.momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }
    best_accuracy = 0.0

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Optimizer: {optimizer_name} | lr={lr} | weight_decay={weight_decay}")
    print(f"Train samples: {len(train_set)} | Test samples: {len(test_set)}")
    print(f"Run directory: {run_dir}")

    for epoch in range(1, args.epochs + 1):
        # 每轮先训练，再在测试集上评估，并保存当前最优模型权重。
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_eval = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["test_loss"].append(test_eval["loss"])
        history["test_accuracy"].append(test_eval["accuracy"])

        if test_eval["accuracy"] > best_accuracy:
            best_accuracy = test_eval["accuracy"]
            torch.save(model.state_dict(), run_dir / "best_model.pt")

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_eval['loss']:.4f} test_acc={test_eval['accuracy']:.4f}"
        )

    final_eval = evaluate(model, test_loader, criterion, device)
    plot_history(history, run_dir)
    plot_confusion_matrix(final_eval["labels"], final_eval["predictions"], run_dir)
    save_results(args, history, final_eval, best_accuracy, run_dir)

    print(f"Best test accuracy: {best_accuracy:.4f}")
    print(f"Final test accuracy: {final_eval['accuracy']:.4f}")
    print(f"Saved metrics and figures to: {run_dir}")


if __name__ == "__main__":
    main()
