from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

def get(test_split = 0.2, batch_size = 1, num_workers = 1, classes = [0,1,2,3,4,5,6,7,8,9]):
    mnist_train = MNIST('MNIST_DATA/', train = True, \
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,), (0.3081,))
        ]), download=True)
    mnist_test = MNIST('MNIST_DATA/', train = False,
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,), (0.3081,))
        ]))

    mnist_train = getClassSubset(mnist_train, classes)
    mnist_test = getClassSubset(mnist_test, classes)

    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(mnist_test,  batch_size=1000, shuffle=True, num_workers=1)

    return mnist_train, mnist_test, train_loader, test_loader


def getClassSubset(dataset, classes):
    idx = dataset.targets == classes[0]
    for c in classes:
        idx += dataset.targets == c
    dataset.targets = dataset.targets[idx]
    dataset.data = dataset.data[idx]
    return dataset