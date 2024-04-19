import random

from torch.utils.data import DataLoader


def get_random_data() -> list[int]:
    """Get a random list of integers"""
    return random.sample(range(1000), 1000)


def get_random_data_loader(batch_size: int) -> DataLoader:
    """Get a random data loader"""
    return DataLoader(
        get_random_data(),
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
    )
