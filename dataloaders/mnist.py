from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

def get(test_split = 0.2, batch_size = 1, num_workers = 1):
    mnist_train  = MNIST('MNIST_DATA/', train = True, \
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,), (0.3081,))
        ]))
    mnist_test   = MNIST('MNIST_DATA/', train = False,
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,), (0.3081,))
        ]))
    train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True, num_workers=1)

    test_loader  = DataLoader(mnist_test,  batch_size=1000, shuffle=True, num_workers=1)

    return mnist_train, mnist_test, train_loader, test_loader
