import torchvision
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import torchvision.transforms as transforms
import os


def LoadData():
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
    )
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainDataSet=torchvision.datasets.CIFAR10(root='./dataset/CIFAR10',download=True,train=True,transform=transform_train)
    testDataSet=torchvision.datasets.CIFAR10(root='./dataset/CIFAR10',download=True,train=False,transform=transform_test)
    return trainDataSet,testDataSet


def plot_loss_and_accuracy(
        loss_history, trainAcc_history, testAcc_history, save_path, suptitle
):
    loss_history = np.array(loss_history)
    trainAcc_history = np.array(trainAcc_history)
    testAcc_history = np.array(testAcc_history)
    print(testAcc_history)
    # 创建一个新的图像
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle(suptitle, fontsize=16)

    # 使用GridSpec来定义子图的布局
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2])

    # 绘制Loss曲线（位于第一行正中间）
    ax1 = plt.subplot(gs[0, :])
    ax1.plot(range(1, len(loss_history) + 1), loss_history, label="Loss", color="blue")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")

    # 绘制Train Accuracy曲线（位于第二行最左边）
    ax2 = plt.subplot(gs[1, 0])
    ax2.plot(
        range(1, len(trainAcc_history) + 1),
        trainAcc_history,
        label="Train Accuracy",
        color="green",
    )
    ax2.set_ylabel("TrainAcc")
    ax2.set_xlabel("Epoch")

    # 绘制Test Accuracy曲线（位于第二行最右边）
    ax3 = plt.subplot(gs[1, 1])
    ax3.plot(
        range(1, len(testAcc_history) + 1),
        testAcc_history,
        label="Test Accuracy",
        color="red",
    )
    ax3.set_ylabel("TestAcc")
    ax3.set_xlabel("Epoch")

    # 调整子图之间的间距
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存图像到指定路径
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved Figure in {save_path}")

def transformTensor2float(loss_history):
    loss_history = [loss_history[i].item() for i in range(len(loss_history))]
    return loss_history



def create_directory_structure(main_folder="test", num_folders=64, subfolders=["net1", "net2", "net3", "net4", "net5", "pic"]):
    """
    创建目录结构函数。

    :param main_folder: 主文件夹名称。
    :type main_folder: str

    :param num_folders: 要创建的数字文件夹数量。
    :type num_folders: int

    :param subfolders: 每个数字文件夹中包含的子文件夹名称列表。
    :type subfolders: list of str
    """
    # 创建主文件夹
    if not os.path.exists(main_folder):
        os.mkdir(main_folder)

    # 创建数字文件夹（0到num_folders-1）
    for digit in range(num_folders):
        digit_folder = os.path.join(main_folder, str(digit))
        if not os.path.exists(digit_folder):
            os.mkdir(digit_folder)

        # 在数字文件夹中创建子文件夹
        for subfolder in subfolders:
            subfolder_path = os.path.join(digit_folder, subfolder)
            if not os.path.exists(subfolder_path):
                os.mkdir(subfolder_path)

    print(f"Directory structure created in '{main_folder}'.")

# 使用示例

