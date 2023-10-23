from ResNetBlock import ResNetBlock
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import *
from tensorboardX import SummaryWriter
from torchviz import make_dot
logger=SummaryWriter(log_dir="data/log")

class ResNet18(nn.Module):
    def __init__(self, ResBlock: ResNetBlock, output_features=10, activeFunc=nn.ReLU()):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.activeFunc = activeFunc
        self.beginLayer = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            self.activeFunc()
        )

        self.layer1 = self.make_layer(ResBlock, 64, 2, 1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, 2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, 2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, 2)
        self.avgPool = nn.AvgPool2d(kernel_size=3)
        self.finalLayer = nn.Sequential(
            nn.Linear(512, output_features)
        )

    def make_layer(self, block: ResNetBlock, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride=stride, activeFunc=self.activeFunc))
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.beginLayer(x)
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.avgPool(out4)
        out6 = self.finalLayer(out5.view(out.size(0), -1))
        return out6,out1,out2,out3,out4,out5

def ResNet18train(model,
                  train_loader,
                  test_loader,
                  criterion,
                  optimizer,
                  num_epochs,
                  learning_rate,
                  batch_size,
                  regularization
                  ):

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
    max_loss = torch.inf
    max_train_acc = 0
    max_test_acc = 0
    print(f"Total Epoch is {num_epochs}")

    # 训练过程
    for epoch in range(1, num_epochs + 1):
        total_test_correct_items = 0
        total_train_correct_items = 0
        total_loss = 0
        total_train_samples = 0
        total_test_samples = 0
        train_progress = tqdm(train_loader, desc=f'epoch {epoch}: Training', leave=False)
        # 遍历训练数据集的小批量
        for inputs, labels in train_progress:
            # 将输入和输出传入GPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 进行模型的训练
            optimizer.zero_grad()
            # Resnet
            # outputs,_,_,_,_,_ = model(inputs)
            # Lenet
            outputs=model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 获取当前输出的标签，同时比较与labels的区别，计算预测正确的情况，同时统计一共计算了多少个样本
            total_train_correct_items, total_train_samples = calculateACC(
                labels, outputs, total_train_correct_items, total_train_samples
            )

            # 统计总损失
            total_loss += loss

        train_progress.close()
        # 切换为测试模式
        model.eval()
        test_progress = tqdm(test_loader, desc=f'epoch {epoch}: Checking', leave=False)
        # 遍历测试数据集的小批量
        for inputs, labels in test_progress:
            # 将输入和输出传入GPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Resnet
            # outputs,_,_,_,_,_ = model(inputs)
            # Lenet
            outputs=model(inputs)
            total_test_correct_items, total_test_samples = calculateACC(
                labels, outputs, total_test_correct_items, total_test_samples
            )
        test_progress.close()
        # 切换为训练模式
        model.train()

        # 计算训练精度和损失
        train_acc = total_train_correct_items / total_train_samples
        train_loss = total_loss / total_train_samples
        test_acc = total_test_correct_items / total_test_samples

        logger.add_scalar("train loss",train_loss,global_step=epoch)
        logger.add_scalar("train accuracy", train_acc, global_step=epoch)
        logger.add_scalar("test accuracy", test_acc, global_step=epoch)

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

        # 生成日志文件内容
        echo_string = f"epoch:{epoch} -> loss:{train_loss} , train_acc:{train_acc},test_acc:{test_acc}"
        print(echo_string)

        # 将训练集精度，测试集精度，训练损失,日志文件打表
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        loss_history.append(train_loss)
        journal_history.append(echo_string)

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
        model: ResNet18,  # PyTorch模型，包括网络结构和参数
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
    save_path = model_path_format.format(
        reason,
        num_epochs,
        batch_size,
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
