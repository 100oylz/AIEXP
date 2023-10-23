import json

from torch import nn as nn, optim as optim
from torch.utils.data import DataLoader

from utils import LoadData, plot_loss_and_accuracy
from MultiSoftMaxNetWork import MultiSoftMaxNetWork, train
from config import *


def MultiSoftMaxNetWorkTrain():
    hidden_features_list = [14 * 14]
    batch_size_list = [1 << i for i in range(6, 10, 2)]
    num_epoch_list = [100]
    learning_rate_list = [0.05, 0.1]
    regularization_list = ["Regularization"]
    # 加载数据
    trainDataSet, testDataSet = LoadData()
    print("LoadData Success!")
    # 28*28,14*14,10
    default_model = MultiSoftMaxNetWork(
        input_features=28 * 28,
        hidden_features=7 * 7,
        output_features=10,
        activeFunc=nn.ReLU(),
    ).to("cuda:0")
    # SGDM
    default_learning_rate = 0.01
    optimizer_list = [
        optim.Adam(
            default_model.parameters(), lr=default_learning_rate, betas=(0.9, 0.999)
        ),
        optim.RMSprop(
            default_model.parameters(),
            lr=default_learning_rate,
            alpha=0.99,
            eps=1e-08,
            momentum=0,
        ),
        optim.Adagrad(default_model.parameters(), lr=default_learning_rate, lr_decay=0),
        optim.Adadelta(default_model.parameters(), rho=0.9, eps=1e-06),
    ]

    # 默认模型
    # MyTrain(trainDataSet, testDataSet)

    """
    # 修正batchsize
    for batch_size in batch_size_list:
        train_loader = DataLoader(
            trainDataSet,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        test_loader = DataLoader(
            testDataSet,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        optimizer = optim.SGD(
            default_model.parameters(), lr=default_learning_rate, momentum=0.9
        )
        MyTrain(
            trainDataSet=trainDataSet,
            testDataSet=testDataSet,
            optimizer=optimizer,
            train_loader=train_loader,
            test_loader=test_loader,
        )

    # 修改隐藏层大小
    for hidden_features in hidden_features_list:
        model = MultiSoftMaxNetWork(
            input_features=28 * 28,
            hidden_features=hidden_features,
            output_features=10,
            activeFunc=nn.ReLU(),
        ).to("cuda:0")
        optimizer = optim.SGD(
            model.parameters(), lr=default_learning_rate, momentum=0.9
        )
        MyTrain(
            trainDataSet=trainDataSet,
            testDataSet=testDataSet,
            optimizer=optimizer,
            model=model,
        )
    """
    # 修改迭代次数
    for num_epoch in num_epoch_list:
        MyTrain(
            trainDataSet=trainDataSet,
            testDataSet=testDataSet,
            # model=default_model,
            # optimizer=optimizer,
            num_epoch=num_epoch,
        )

    """
    # 修改学习率
    for learning_rate in learning_rate_list:
        optimizer = optim.SGD(
            default_model.parameters(), lr=learning_rate, momentum=0.9
        )
        MyTrain(
            trainDataSet=trainDataSet,
            testDataSet=testDataSet,
            optimizer=optimizer,
            learning_rate=learning_rate,
        )
    # 正则化
    for regularization_list in regularization_list:
        # 加入权重衰减，L2正则化
        optimizer = optim.SGD(
            default_model.parameters(),
            lr=default_learning_rate,
            momentum=0.9,
            weight_decay=0.0001,
        )
        MyTrain(
            trainDataSet=trainDataSet,
            testDataSet=testDataSet,
            optimizer=optimizer,
            regularization=regularization_list,
        )
    # 修改优化器
    for optimizer in optimizer_list:
        MyTrain(trainDataSet=trainDataSet, testDataSet=testDataSet, optimizer=optimizer)
    """


def MyTrain(
    trainDataSet,
    testDataSet,
    criterion=None,
    batch_size=None,
    learning_rate=None,
    model=None,
    num_epoch=None,
    optimizer=None,
    regularization=None,
    train_loader=None,
    test_loader=None,
):
    # 如果参数没有传递，使用默认值
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if batch_size is None:
        batch_size = 1 << 4
    if learning_rate is None:
        learning_rate = 0.01
    if model is None:
        model = MultiSoftMaxNetWork(
            input_features=28 * 28,
            hidden_features=7 * 7,
            output_features=10,
            activeFunc=nn.ReLU(),
        ).to("cuda:0")
    if num_epoch is None:
        num_epoch = 50
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    if regularization is None:
        regularization = "NoRegularization"
    if train_loader is None:
        train_loader = DataLoader(
            trainDataSet,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
    if test_loader is None:
        test_loader = DataLoader(
            testDataSet,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    loss_history, trainAcc_history, testAcc_history, journal_history = train(
        model=model,
        train_loader=test_loader,
        test_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epoch,
        learning_rate=learning_rate,
        batch_size=batch_size,
        regularization=regularization,
    )
    loss_history = transformTensor2float(loss_history)
    save_path = png_path_format.format(
        num_epoch,
        batch_size,
        model.hidden_features,
        "(" + str(learning_rate).replace(".", "_") + ")",
        optimizer.__class__.__name__,
        regularization,
    )
    title = "Training renderings"
    plot_loss_and_accuracy(
        loss_history,
        trainAcc_history,
        testAcc_history,
        save_path=save_path,
        suptitle=title,
    )
    journal_path = journal_path_format.format(
        num_epoch,
        batch_size,
        model.hidden_features,
        "(" + str(learning_rate).replace(".", "_") + ")",
        optimizer.__class__.__name__,
        regularization,
    )
    with open(journal_path, "w", encoding="utf8") as f:
        json.dump(journal_history, f, ensure_ascii=False)
        f.flush()
    print(f"Saved Journal in {journal_path}")


def transformTensor2float(loss_history):
    loss_history = [loss_history[i].item() for i in range(len(loss_history))]
    return loss_history


if __name__ == "__main__":
    MultiSoftMaxNetWorkTrain()
