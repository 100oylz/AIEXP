import json

import numpy as np
import torch.utils.data
from torchviz import make_dot
import torchvision
from ResNet18 import *
from config import *
from utils import *
import seaborn as sns
from sklearn.metrics import confusion_matrix

def baseTrain(batch_size=64, num_workers=8, learning_rate=0.01, num_epochs=100, regularization='NoRegularization'):
    trainDataset, testDataset = LoadData()
    print("Load DataSet Success")
    train_loader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers)

    criterion = nn.CrossEntropyLoss()
    model = ResNet18(ResNetBlock, 10, nn.PReLU).to('cuda')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99)

    loss_history, trainAcc_history, testAcc_history, journal_history = ResNet18train(model,
                                                                                     train_loader, test_loader,
                                                                                     criterion, optimizer, num_epochs,
                                                                                     learning_rate, batch_size,
                                                                                     regularization)
    loss_history = transformTensor2float(loss_history)

    png_path = png_path_format.format(
        num_epochs,
        batch_size,
        "(" + str(learning_rate).replace(".", "_") + ")",
        optimizer.__class__.__name__,
        regularization,
    )
    journal_path = journal_path_format.format(
        num_epochs,
        batch_size,
        "(" + str(learning_rate).replace(".", "_") + ")",
        optimizer.__class__.__name__,
        regularization,
    )
    title = "Training renderings"
    plot_loss_and_accuracy(loss_history, trainAcc_history, testAcc_history, save_path=png_path, suptitle=title)

    with open(journal_path, "w", encoding="utf8") as f:
        json.dump(journal_history, f, ensure_ascii=False)

    print(f"Saved Journal in {journal_path}")


def test(batch_size=64, num_workers=8):
    create_directory_structure()
    print('create Dirctionary Success!')
    trainDataset, testDataset = LoadData()
    print("Load DataSet Success")
    test_loader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers)
    model = ResNet18(ResNetBlock, 10, nn.PReLU).to('cuda')
    modelstatedict = torch.load('checkpoint/CIFAR10_TrainAcc_100_64_(0_01)_NoRegularization_type.pt')
    model.load_state_dict(modelstatedict)
    model.eval()

    def showGrayImage(image: torch.Tensor, netName: str, labelName: str):
        print(netName)
        batch_size, num_channels, height, width = image.size()

        if image.is_cuda:
            image = image.cpu().detach()

        for i in range(batch_size):
            for j in range(num_channels):
                img = image[i, j].numpy()

                imgfilePath = test_png_format.format(i, netName, classes[labelName[i]], j)
                plt.imsave(imgfilePath, img, cmap='gray')

    def showColorImage(image: torch.Tensor, netName: str, labelName: str):
        print(netName)
        batch_size, num_channels, height, width = image.size()

        if image.is_cuda:
            image = image.cpu()

        for i in range(batch_size):
            img = image[i].permute(1, 2, 0).detach().numpy()
            img = np.clip(img, 0, 1)
            imgfilePath = test_png_format.format(i, netName, classes[labelName[i]], 0)  # 0 表示通道索引
            plt.imsave(imgfilePath, img, cmap='gray')
    predicted_class_list=[]
    true_class_list=[]
    for image, labels in test_loader:
        image = image.to('cuda')
        labels = labels.to('cuda')
        outputs, out1, out2, out3, out4, out5 = model(image)
        _, predicted = torch.max(outputs, dim=1)  # 第一个参数为是当前张量的最大值的数值，第二个参数是当前张量最大值的索引
        true_class_list.extend(labels.cpu().numpy())
        predicted_class_list.extend(predicted.cpu().numpy())
    # 计算混淆矩阵
    cm = confusion_matrix(true_class_list, predicted_class_list)

    # 创建图表
    plt.figure(figsize=(len(classes) + 2, len(classes) + 2))
    sns.set(font_scale=1.2)  # 设置字体大小
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False, square=True,
                xticklabels=classes,
                yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title("Confusion Matrix")
    plt.savefig(f'test_confusion_matrix.png')



def netPng():
    model = ResNet18(ResNetBlock, 10, nn.PReLU)
    x = torch.zeros([1, 3, 32, 32])
    y = model(x)
    g = make_dot(y)
    g.view()


if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = (128, 128)
    # baseTrain(num_epochs=1)
    test()
    # netPng()
