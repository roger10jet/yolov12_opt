import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ultralytics.cfg import TASK2DATA, TASKS

from ultralytics.nn.extramodules import ESNetV3, MSAM


TASK_MODEL_DATA = [(TASK2DATA[task]) for task in TASKS]

def test_msam():
    # 创建MSAM模块实例
    channels = 64
    msam = MSAM(channels)

    # 生成测试输入
    batch_size = 2
    height, width = 32, 32
    x = torch.randn(batch_size, channels, height, width)

    print(f"输入特征图尺寸: {x.shape}")

    # 前向传播
    with torch.no_grad():
        output = msam(x)

    print(f"输出特征图尺寸: {output.shape}")
    print("MSAM模块测试完成！")

def test_esnetv3(data):
    """测试ESNetV3模型"""
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集（示例使用CIFAR-10）
    train_dataset = datasets.CIFAR10(root="./data", train=True,
                                     download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32,
                              shuffle=True, num_workers=2)

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESNetV3(num_classes=10, width_multiplier=1.0).to(device)

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # 训练循环
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

        scheduler.step()
        accuracy = 100. * correct / total
        print(f'Epoch {epoch} completed. Loss: {running_loss / len(train_loader):.4f}, '
              f'Accuracy: {accuracy:.2f}%')

    # 保存模型
    torch.save(model.state_dict(), 'esnetv3_final.pth')
    print("测试完成，模型已保存！")
