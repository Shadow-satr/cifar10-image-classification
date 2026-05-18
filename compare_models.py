import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args():
    # 统一设置三种模型的公共训练参数，便于保证对比实验公平。
    parser = argparse.ArgumentParser(description="Run CIFAR-10 MLP/CNN/ResNet18 experiments.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--subset-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--strong-augment", action="store_true")
    parser.add_argument("--resnet-epochs", type=int, default=0, help="If >0, override epochs for ResNet18 only.")
    parser.add_argument("--resnet-batch-size", type=int, default=0, help="If >0, override batch size for ResNet18 only.")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    return parser.parse_args()


def newest_metrics(output_dir, model):
    # 训练脚本每次会创建新目录，这里取指定模型最新一次运行的指标文件。
    candidates = sorted(output_dir.glob(f"*_{model}/metrics.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        return None
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


def run_model(args, model):
    # 通过子进程调用 train_cifar10.py，让三个模型复用完全相同的训练与评估流程。
    epochs = args.resnet_epochs if model == "resnet18" and args.resnet_epochs > 0 else args.epochs
    batch_size = args.resnet_batch_size if model == "resnet18" and args.resnet_batch_size > 0 else args.batch_size
    command = [
        sys.executable,
        "train_cifar10.py",
        "--model",
        model,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(args.num_workers),
        "--optimizer",
        args.optimizer,
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--dropout",
        str(args.dropout),
        "--label-smoothing",
        str(args.label_smoothing),
        "--output-dir",
        str(args.output_dir),
    ]
    if args.subset_size:
        command.extend(["--subset-size", str(args.subset_size)])
    if args.smoke_test:
        command.append("--smoke-test")
    if args.strong_augment:
        command.append("--strong-augment")

    subprocess.run(command, check=True)
    return newest_metrics(args.output_dir, model)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for model in ("mlp", "cnn", "resnet18"):
        # 依次训练 MLP、普通 CNN 和 ResNet18，并汇总每个模型的测试集表现。
        metrics = run_model(args, model)
        summary[model] = {
            "best_test_accuracy": metrics["best_test_accuracy"],
            "final_test_accuracy": metrics["final_test_accuracy"],
            "final_test_loss": metrics["final_test_loss"],
        }

    summary_path = args.output_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nModel comparison")
    print("----------------")
    for model, metrics in summary.items():
        print(
            f"{model.upper():>8}: "
            f"best_acc={metrics['best_test_accuracy']:.4f}, "
            f"final_acc={metrics['final_test_accuracy']:.4f}, "
            f"final_loss={metrics['final_test_loss']:.4f}"
        )
    print(f"\nSaved comparison summary to {summary_path}")


if __name__ == "__main__":
    main()


