import math

import torch
import torch.nn as nn
from config import *


class MultiSoftMaxNetWork(nn.Module):
    def __init__(self, input_features, hidden_features, output_features, activeFunc):
        """
        初始化网络结构

        :param input_features: 输入特征数
        :type input_features: int
        :param hidden_features: 隐藏层特征数
        :type hidden_features: int
        :param output_features: 输出特征数
        :type output_features: int
        :param activeFunc: 激活函数
        :type activeFunc:torch.nn.Module
        """
        super(MultiSoftMaxNetWork, self).__init__()

        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_features = output_features

        self.linear1 = nn.Linear(input_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, output_features)
        self.activeFunc = activeFunc

        self.model = nn.Sequential(self.linear1, self.activeFunc, self.linear2)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


def train(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    num_epochs,
    learning_rate: float,
    batch_size: int,
    regularization: str,
):
    """
    训练模型。

    :param model: PyTorch模型，包括网络结构和参数。
    :type model: MultiSoftMaxNetWork

    :param train_loader: 用于训练的数据加载器。
    :type train_loader: DataLoader

    :param test_loader: 用于测试/验证的数据加载器。
    :type test_loader: DataLoader

    :param criterion: 损失函数，用于计算模型预测与实际值之间的损失。
    :type criterion: nn.Module

    :param optimizer: 优化器，用于更新模型参数以减小损失。
    :type optimizer: torch.optim.Optimizer

    :param num_epochs: 训练的总轮次，控制训练过程的迭代次数。
    :type num_epochs: int

    :param learning_rate: 学习率，控制优化器的学习步长。
    :type learning_rate: float

    :param batch_size: 每个小批量训练的样本数。
    :type batch_size: int

    :param regularization: 正则化类型，用于控制正则化的方式（例如 "NoRegularization" 表示没有正则化）。
    :type regularization: str

    :return:
        - loss_history: 包含每个训练轮次的损失值的列表。
        :rtype: list of float

        - train_acc_history: 包含每个训练轮次的训练准确率的列表。
        :rtype: list of float

        - loss_history: 包含每个训练轮次的验证/测试损失值的列表。
        :rtype: list of float

        - journal_history: 包含每个训练轮次的日志信息的列表，例如保存模型的日志。
        :rtype: list of str
    """
    # 检查是否支持CUDA，如果支持则使用GPU加速
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Use Cuda!")
    else:
        print("Use Cpu!")

    # 将模型移动到GPU上（如果可用）
    model.to(device)

    # 初始化训练过程中的列表
    train_acc_history = []
    test_acc_history = []
    loss_history = []
    journal_history = []
    max_loss = math.inf
    max_train_acc = 0
    max_test_acc = 0

    # 训练过程
    for epoch in range(1, num_epochs + 1):
        total_test_correct_items = 0
        total_train_correct_items = 0
        total_loss = 0
        total_train_samples = 0
        total_test_samples = 0

        # 遍历训练数据集的小批量
        for inputs, labels in train_loader:
            # 将输入和输出传入GPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 进行模型的训练
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # loss.backward()
            optimizer.step()

            # 获取当前输出的标签，同时比较与labels的区别，计算预测正确的情况，同时统计一共计算了多少个样本
            total_train_correct_items, total_train_samples = calculateACC(
                labels, outputs, total_train_correct_items, total_train_samples
            )

            # 统计总损失
            total_loss += loss
            print(loss)

        # 切换为测试模式
        model.eval()

        # 遍历测试数据集的小批量
        for inputs, labels in test_loader:
            # 将输入和输出传入GPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            total_test_correct_items, total_test_samples = calculateACC(
                labels, outputs, total_test_correct_items, total_test_samples
            )

        # 切换为训练模式
        model.train()

        # 计算训练精度和损失
        train_acc = total_train_correct_items / total_train_samples
        train_loss = total_loss / total_train_samples
        test_acc = total_test_correct_items / total_test_samples

        # 生成日志文件内容
        echo_string = f"epoch:{epoch} -> loss:{train_loss} , train_acc:{train_acc},test_acc:{test_acc}"
        print(echo_string)

        # 保存策略
        max_loss = save_dict(
            model,
            train_loss,
            max_loss,
            "Loss",
            epoch,
            journal_history,
            learning_rate,
            batch_size,
            regularization,
            num_epochs,
            optimizer.__class__.__name__,
        )
        max_train_acc = save_dict(
            model,
            train_acc,
            max_train_acc,
            "TrainAcc",
            epoch,
            journal_history,
            learning_rate,
            batch_size,
            regularization,
            num_epochs,
            optimizer.__class__.__name__,
        )
        max_test_acc = save_dict(
            model,
            test_acc,
            max_test_acc,
            "TestAcc",
            epoch,
            journal_history,
            learning_rate,
            batch_size,
            regularization,
            num_epochs,
            optimizer.__class__.__name__,
        )

        # 将训练集精度，测试集精度，训练损失,日志文件打表
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        loss_history.append(train_loss)
        journal_history.append(echo_string)

    print(
        f"Train {model.input_features}-{model.hidden_features}-{model.output_features} Finished!"
    )
    journal = f"The Best Loss:{max_loss}!\nThe Best Train Accuracy:{max_train_acc}!\nThe Best Test Accuract:{max_test_acc}!"
    print(journal)
    journal_history.append(journal)
    return loss_history, train_acc_history, test_acc_history, journal_history


def calculateACC(labels, outputs, total_train_correct_items, total_train_samples):
    _, predicted = torch.max(outputs, dim=1)  # 第一个参数为是当前张量的最大值的数值，第二个参数是当前张量最大值的索引
    total_train_correct_items += (predicted == labels).sum().item()
    total_train_samples += labels.size(0)
    return total_train_correct_items, total_train_samples


def save_dict(
    model: MultiSoftMaxNetWork,  # PyTorch模型，包括网络结构和参数
    content: float,  # 要比较的内容值
    threshold: float,  # 阈值，与content比较以确定是否保存模型
    reason: str,  # 原因字符串，用于决定是否保存模型
    epoch: int,  # 当前训练的轮次（epoch）
    journal_list: list,  # 存储日志的列表
    learning_rate: float,
    batch_size: int,
    regularization: str,
    num_epochs: int,
    optimzerName: str,
):
    """
    根据给定的条件保存模型参数字典。

    :param model: PyTorch模型，包括网络结构和参数
    :type model: MultiSoftMaxNetWork

    :param content: 要比较的内容值
    :type content: float

    :param threshold: 阈值，与content比较以确定是否保存模型
    :type threshold: float

    :param reason: 原因字符串，用于决定是否保存模型（例如 "loss" 或其他原因）
    :type reason: str

    :param epoch: 当前训练的轮次（epoch）
    :type epoch: int

    :param journal_list: 存储日志的列表
    :type journal_list: list

    :return: 更新后的阈值
    :rtype: float
    """
    save_path = save_dict_path_format.format(
        reason,
        num_epochs,
        batch_size,
        model.input_features,
        model.hidden_features,
        model.output_features,
        "(" + str(learning_rate).replace(".", "_") + ")",
        regularization,
        model.activeFunc.__class__.__name__,
        optimzerName,
    )
    if reason == "Loss":
        if content < threshold:
            threshold = content
            torch.save(model.state_dict(), save_path)
            journal = f"epoch {epoch} saved! Because of {reason}"
            print(journal)
            journal_list.append(journal)
    else:
        if content > threshold:
            threshold = content
            torch.save(model.state_dict(), save_path)
            journal = f"epoch {epoch} saved! Because of {reason}"
            print(journal)
            journal_list.append(journal)
    return threshold
