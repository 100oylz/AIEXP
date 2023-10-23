import torch
import torch.nn as nn
from utils import *
from config import *
import json
import torch.optim as optim
from ResNet18 import ResNet18train
class Lenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.activeFunc=nn.ReLU()
        self.fc = nn.Sequential(nn.Linear(16 * 5 * 5, 120), nn.Linear(120, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.fc(x.view(x.shape[0], -1))
        return out

def baseTrain(batch_size=64, num_workers=8, learning_rate=0.01, num_epochs=100, regularization='NoRegularization'):
    trainDataset, testDataset = LoadData()
    print("Load DataSet Success")
    train_loader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers)

    criterion = nn.CrossEntropyLoss()
    model = Lenet().to('cuda')
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

if __name__ == '__main__':
    baseTrain(num_epochs=50,learning_rate=1e-4,batch_size=256,num_workers=1)
