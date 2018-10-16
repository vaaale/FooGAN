import torch
import torch.utils.data as data
import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, transform=None, categories=[]):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        catIds = self.coco.getCatIds(catNms=categories)
        self.ids = [key for key, val in self.coco.anns.items() if val['category_id'] in catIds]
        # self.ids = list(self.coco.anns.keys())
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        ann_id = self.ids[index]
        img_id = coco.anns[ann_id]['image_id']
        img_cat = coco.anns[ann_id]['category_id']
        img_bbox = coco.anns[ann_id]['bbox']
        images = coco.loadImgs(img_id)[0]
        path = images['file_name']


        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return np.array(image).transpose([1, 2, 0])

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths


def get_loader(root, json, transform, batch_size, shuffle, num_workers, categories):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json,
                       transform=transform,
                       categories=categories)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)
    return data_loader


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import skimage.io as io

    image_dir = 'z:/Dataset/coco/val/orig/val2017/'
    ann_file = 'z:/Dataset/coco/annotations/instances_val2017.json'

    def toGray(img):
        result = np.zeros((*img.size, 3))
        gray = img.convert("L")
        result[:,:, 0] = gray
        result[:,:, 1] = gray
        result[:,:, 2] = gray

        result = np.concatenate([img, result], 1)
        return result

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.Lambda(toGray),
        transforms.ToTensor(),
    ])

    dataloader = get_loader(root = image_dir, json=ann_file, batch_size=4, shuffle=True, num_workers=1, transform=tf, categories=['truck'])

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched.size())
        plt.imshow(sample_batched[0])
        plt.show()
        io.imsave('../trash/img_{}.png'.format(i_batch), sample_batched[0] / 255)
        if i_batch > 4:
            break
