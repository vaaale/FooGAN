import torch
import torch.utils.data as data
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder


def get_folder_loader(dataroot, transform, batch_size, shuffle, num_workers):
    dataset = ImageFolder(root=dataroot,
                               transform=transform, target_transform=None)
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=num_workers)

    return dataloader



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


def get_coco_loader(root, json, transform, batch_size, shuffle, num_workers, categories):
    from pycocotools.coco import COCO
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

def toGray(img):
    result = np.zeros((*img.size, 3))
    gray = img.convert("L")
    result[:,:, 0] = gray
    result[:,:, 1] = gray
    result[:,:, 2] = gray

    result = np.concatenate([img, result], 1)
    toTensor = transforms.ToTensor()
    return {'A': toTensor(img), 'B': toTensor(result/255)}


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import skimage.io as io

    image_dir = 'z:/Dataset/coco/val/orig/'
    # ann_file = 'z:/Dataset/coco/annotations/instances_val2017.json'


    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.Lambda(toGray),
        # transforms.ToTensor(),
    ])

    # dataloader = get_coco_loader(root = image_dir, json=ann_file, batch_size=4, shuffle=True, num_workers=1, transform=tf, categories=['truck'])
    dataloader = get_folder_loader(dataroot=image_dir, transform=tf, batch_size=4, shuffle=True, num_workers=1)
    for i_batch, (data, labels) in enumerate(dataloader):
        A = data['A']
        B = data['B']
        print(i_batch, A.size())
        plt.imshow(A[0].numpy().transpose(1, 2, 0))
        plt.imshow(B[0].numpy().transpose(1, 2, 0))
        plt.show()
        io.imsave('trash/img_A_{}.png'.format(i_batch), A[0].numpy().transpose(1, 2, 0))
        io.imsave('trash/img_B_{}.png'.format(i_batch), B[0].numpy().transpose(1, 2, 0))
        if i_batch > 4:
            break
