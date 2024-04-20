"""Load MNIST data from torchvision.datasets.MNIST."""
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor


def get_mnist_data_loader(batch_size: int) -> DataLoader:
    """Get MNIST data loader."""
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    mnist = MNIST(root='data', train=True, download=True, transform=transform)
    return DataLoader(mnist, batch_size=batch_size, shuffle=True)
