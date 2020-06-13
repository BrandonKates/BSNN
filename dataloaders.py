from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, MNIST
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
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    ])

    trainset = CIFAR10(root='./CIFAR10_DATA', 
            train=True, download=True,
            transform=transform )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers)

    testset = CIFAR10(root='./CIFAR10_DATA', 
            train=False, download=True,
            transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers)

    return trainset, testset, trainloader, testloader
