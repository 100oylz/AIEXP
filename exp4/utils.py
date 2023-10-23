import torchvision
import torchvision.transforms as transform


def loadData():
    train_transform = transform.Compose(
        [
            transform.ToTensor(),
            transform.Pad(4),
            transform.RandomCrop(28),
            transform.RandomRotation(5),
        ]
    )
    test_transform = transform.Compose(
        [
            transform.ToTensor(),
        ]
    )

    trainData = torchvision.datasets.CIFAR10(
        './CIFAR10', download=True, train=True, transform=train_transform)
    testData = torchvision.datasets.CIFAR10(
        './CIFAR10', download=True, train=False, transform=test_transform)
    return trainData,testData
