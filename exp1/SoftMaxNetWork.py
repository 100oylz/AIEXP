import torch
import torch.nn as nn
from config import *


class SoftMaxNetWork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SoftMaxNetWork, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        """

        :param x: (batch_size,1*28*28)
        :type x: torch.Tensor
        :return:(bathc_size,num_classes)
        :rtype: torch.Tensor
        """
        x = x.view(x.size(0), -1)
        y = self.linear(x)
        return y


def train(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    num_epochs,
    max_ensure_loss=MAX_ENSURE_LOSS,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Use Cuda!")
    else:
        print("Use Cpu!")
    model.to(device)
    acc_history = []
    loss_history = []
    max_loss = torch.inf
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
        model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs, dim=1)

                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        acc_history.append(accuracy)
        epoch_loss = total_loss / num_batches
        loss_history.append(epoch_loss)
        print(f"epoch {epoch}: Loss:{epoch_loss}")
        if epoch_loss < max_loss:
            max_loss = epoch_loss
            torch.save(model.state_dict(), ModelSavePath)
            print(f"epoch {epoch} Saved! Loss:{epoch_loss}")

        if epoch_loss < max_ensure_loss:
            break
        model.train()
    print("Train Finished!")
    return loss_history, acc_history
