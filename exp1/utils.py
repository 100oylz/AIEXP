import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from config import *

import numpy as np


def LoadData():
    mnist_train = torchvision.datasets.FashionMNIST(
        root=DATASET_FASHION_MNIST,
        train=True,
        download=False,
        transform=transforms.ToTensor(),
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root=DATASET_FASHION_MNIST,
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )
    return mnist_train, mnist_test


def loadLabel(labelNum):
    return [LABEL_LIST[labelNum[i]] for i in range(labelNum)]


def saveFigurePng(figurehistorydata, filepath, title, xlabel, ylabel):
    y = figurehistorydata
    x = [i + 1 for i in range(len(figurehistorydata))]
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filepath)
    plt.show()


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
    # plt.show()


if __name__ == "__main__":
    mnist_train, mnist_test = LoadData()
    print(mnist_train[0])
