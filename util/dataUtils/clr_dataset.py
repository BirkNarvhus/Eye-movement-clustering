"""
This file contains the code for the clr dataset.
"""

from typing import Optional, Callable

from PIL import Image
from torchvision import datasets


class ClrDataset(datasets.MNIST):
    """
    Clr dataset class for contrastive learning.
    Extends the MNIST dataset class.

    """
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        applies the transformation to the items.
        :param idx: index of the item
        :return: transformed items and target
        """
        img, target = self.data[idx], int(self.targets[idx])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            x_i = self.transform(img)
            x_j = self.transform(img)
        else:
            raise ValueError("Transform needs to be set for clr dataset.")

        if self.target_transform is not None:
            target = self.target_transform(target)

        return x_i, x_j, target
