from torch import nn as nn, optim as optim
from torch.utils.data import DataLoader

from utils import LoadData, saveFigurePng
from SoftMaxNetWork import SoftMaxNetWork, train
from config import *


def SoftMaxNetWorkTrain():
    model = SoftMaxNetWork(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    trainDataSet, testDataSet = LoadData()
    print("LoadData Success!")
    train_loader = DataLoader(
        trainDataSet, batch_size=BATCHSIZE, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        testDataSet, batch_size=BATCHSIZE, shuffle=False, num_workers=num_workers
    )
    num_epochs = MAX_NUM_EPOCH
    loss_history, acc_history = train(
        model, train_loader, test_loader, criterion, optimizer, num_epochs
    )
    print("Train Finished")
    saveFigurePng(loss_history, LossPngFilePath, LossPngChartTitle, "epoch", "Loss")
    saveFigurePng(acc_history, AccPngFilePath, AccPngChartTitle, "epoch", "ACC")


if __name__ == "__main__":
    SoftMaxNetWorkTrain()
