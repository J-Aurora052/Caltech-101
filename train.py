import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset import get_dataloaders
from models import FineTunedResNet18
import argparse
import time

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 20 == 0 and writer is not None:
            writer.add_scalar('batch/train_loss', loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('batch/train_acc', 100. * correct / total, epoch * len(train_loader) + batch_idx)

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total

    if writer is not None:
        writer.add_scalar('epoch/train_loss', train_loss, epoch)
        writer.add_scalar('epoch/train_acc', train_acc, epoch)

    return train_loss, train_acc


def validate(model, val_loader, criterion, epoch, writer=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total

    if writer is not None:
        writer.add_scalar('epoch/val_loss', val_loss, epoch)
        writer.add_scalar('epoch/val_acc', val_acc, epoch)

    return val_loss, val_acc


def run_experiment(config, experiment_name):
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    log_dir = os.path.join(config['log_dir'], experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    # 数据加载
    train_loader, val_loader, classes = get_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size']
    )
    print(f"Training on {len(classes)} classes")

    # 模型初始化
    model = FineTunedResNet18(
        num_classes=len(classes),
        pretrained=config.get('pretrained', True)
    ).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    # 设置不同层的学习率
    if config.get('pretrained', True):
        optimizer = optim.SGD([
            {'params': model.get_pretrained_params(), 'lr': config['lr_pretrained']},
            {'params': model.get_new_params(), 'lr': config['lr_new']}
        ], momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['lr_new'],
            momentum=0.9,
            weight_decay=1e-4
        )

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # TensorBoard记录
    writer = SummaryWriter(log_dir)

    # 训练循环
    best_acc = 0.0
    start_time = time.time()

    print(f"\nStarting Experiment: {experiment_name}")
    print(f"Using {'pretrained' if config.get('pretrained', True) else 'random initialized'} model")
    print(f"Batch size: {config['batch_size']}, Epochs: {config['epochs']}")
    print(f"Learning rates - New: {config['lr_new']}, Pretrained: {config.get('lr_pretrained', 'N/A')}")

    for epoch in range(config['epochs']):
        epoch_start = time.time()

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, writer
        )

        # 验证
        val_loss, val_acc = validate(
            model, val_loader, criterion, epoch, writer
        )

        # 更新学习率
        scheduler.step()

        # 记录当前学习率
        for i, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar(f'lr/group_{i}', param_group['lr'], epoch)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config['save_dir'], f'best_{experiment_name}.pth'))

        # 打印信息
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch + 1}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s")

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(config['save_dir'], f'final_{experiment_name}.pth'))

    total_time = time.time() - start_time
    print(f"\nExperiment {experiment_name} completed in {total_time / 60:.1f} minutes")
    print(f"Best validation accuracy: {best_acc:.2f}%")

    writer.close()
    return best_acc


def main():
    parser = argparse.ArgumentParser(description='Caltech-101 Classification')
    parser.add_argument('--data_dir', type=str, default='./caltech101', help='Dataset path')
    parser.add_argument('--save_dir', type=str, default='./results', help='Model save directory')
    parser.add_argument('--log_dir', type=str, default='./logs', help='TensorBoard log directory')
    args = parser.parse_args()

    # 基础配置
    base_config = {
        'data_dir': args.data_dir,
        'save_dir': args.save_dir,
        'log_dir': args.log_dir,
        'batch_size': 32,
        'epochs': 30,  # 固定为30个epoch
        'lr_new': 0.01,
        'lr_pretrained': 0.001
    }

    # 运行标准预训练模型实验
    pretrained_config = {**base_config, 'pretrained': True}
    pretrained_acc = run_experiment(pretrained_config, "pretrained_standard")

    # 运行随机初始化对比实验
    scratch_config = {
        **base_config,
        'pretrained': False,
        'lr_new': 0.1  # 随机初始化需要更大的学习率
    }
    scratch_acc = run_experiment(scratch_config, "random_initialized")

    # 运行学习率对比实验
    lr_experiments = [
        {'name': 'high_lr', 'lr_new': 0.1, 'lr_pretrained': 0.01},
        {'name': 'medium_lr', 'lr_new': 0.01, 'lr_pretrained': 0.001},
        {'name': 'low_lr', 'lr_new': 0.001, 'lr_pretrained': 0.0001}
    ]

    lr_results = {}
    for exp in lr_experiments:
        config = {
            **base_config,
            'lr_new': exp['lr_new'],
            'lr_pretrained': exp['lr_pretrained']
        }
        lr_results[exp['name']] = run_experiment(config, f"lr_{exp['name']}")

    # 打印汇总结果
    print("\n=== Final Results Summary ===")
    print(f"Pretrained model accuracy: {pretrained_acc:.2f}%")
    print(f"Random initialized accuracy: {scratch_acc:.2f}%")
    print("\nLearning rate experiments:")
    for name, acc in lr_results.items():
        print(f"{name}: {acc:.2f}%")


if __name__ == '__main__':
    main()