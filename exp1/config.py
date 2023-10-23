LABEL_LIST = [
    "t-shirt",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle boot",
]
DATASET_FASHION_MNIST = "./dataset/FashionMNIST"

ModelSavePath = "./checkpoint/MNIST_best.pt"
input_size = 784
num_classes = 10
num_workers = 4
BATCHSIZE = 64

MAX_ENSURE_LOSS = 1e-2
MAX_NUM_EPOCH = 100
LossPngFilePath = "Loss.png"
AccPngFilePath = "Accuracy.png"
LossPngChartTitle = "Loss Change During Train Chart"
AccPngChartTitle = "Accuracy Change During Train Chart"

save_dict_path_format = "checkpoint/MNIST_{}_best_model_{}_{}_{}_{}_{}_{}_{}_{}_{}.pt"
png_path_format = "Pic/MNIST_{}_{}_{}_{}_{}_{}.png"
journal_path_format = "journal/MNIST_{}_{}_{}_{}_{}_{}.json"
