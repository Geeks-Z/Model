import timm
import torch
import torchvision
from torch import nn, optim
from torchvision import datasets, transforms
import os


def main():
    # 1. 设置设备
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 加载并预处理CIFAR-100数据集
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ViT期望的输入尺寸
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    trainset = torchvision.datasets.CIFAR100(root='/home/team/zhaohongwei/Dataset', train=True,
                                             download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True)

    testset = torchvision.datasets.CIFAR100(root='/home/team/zhaohongwei/Dataset', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False)

    model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True,
                                          num_classes=100).to(device).eval()

    # 4. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. 训练模型
    for epoch in range(20):  # 遍历数据集多次
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:  # 每200个批次打印一次
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # 6. 评估模型
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

if __name__ == '__main__':
    main()
