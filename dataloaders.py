from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, MNIST, SVHN
from torchvision import transforms

def mnist(resize=False, test_split=0.2, batch_size=1, num_workers=1):
    mnist_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
    if resize:
        mnist_transforms = [transforms.Resize(32)] + mnist_transforms
    mnist_transforms = transforms.Compose(mnist_transforms)

    mnist_train = MNIST('MNIST_DATA/', train=True,
            transform=mnist_transforms, download=True)
    mnist_test = MNIST('MNIST_DATA/', train = False,transform=mnist_transforms)

    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(mnist_test,  batch_size=1000, shuffle=True, num_workers=num_workers)

    return mnist_train, mnist_test, train_loader, test_loader


def cifar10(batch_size, num_workers=1):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.247, 0.243, 0.261]
    )

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    trainset = CIFAR10(root='./CIFAR10_DATA', train=True, download=True,
                        transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers)

    testset = CIFAR10(root='./CIFAR10_DATA', train=False, download=True,
                        transform=test_transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers)

    return trainset, testset, trainloader, testloader


def svhn(batch_size, num_workers=1):
    ts = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.4377, .4438, .4782],
                          std=[.1282, .1315, .1123])])
    trainset = SVHN('./SVHN', transform=ts, download=True)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers)
    testset = SVHN('./SVHN', split='test', download=True, transform=ts)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers)

    return trainset, testset, trainloader, testloader

