import torch

from yolov12.ultralytics.nn.extramodules import ESNetV3, MSAM

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

def test_esnetv3():
    model = ESNetV3()

    # 生成测试输入
    channels = 128
    batch_size = 2
    height, width = 64, 64
    x = torch.randn(batch_size, channels, height, width)
    print(f"输入特征图尺寸: {x.shape}")

    # 前向传播
    with torch.no_grad():
        output = model(x)

    print(f"输出特征图尺寸: {output.shape}")
    print("ESNetv3模块测试完成！")
