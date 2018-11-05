import torch
import torch.utils.data as data
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.utils as vutils
import numpy as np
from imageutils import distort, toGray


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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import skimage.io as io

    # image_dir = 'C:/Users/alex/Dataset/CelebA'
    # output_dir = 'C:/Users/alex/Dataset/CelebA_combined/'

    image_dir = 'Y:/Dataset/B'
    output_dir = 'Y:/Dataset/AB_Gray'
    # ann_file = 'z:/Dataset/coco/annotations/instances_val2017.json'

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.Lambda(distort),
    ])

    counter = 1
    dataloader = get_folder_loader(dataroot=image_dir, transform=tf, batch_size=4, shuffle=False, num_workers=1)
    nb_images = len(dataloader)
    train_split = int(nb_images*0.8)
    for i_batch, (data, labels) in enumerate(dataloader):
        A = data['A']
        B = data['B']

        # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        # ax[0].imshow(A[0].numpy().transpose(1, 2, 0))
        # ax[1].imshow(B[0].numpy().transpose(1, 2, 0))
        # plt.savefig('trash/sample_{}.png'.format(i_batch))
        # plt.show()

        # C = torch.cat([B, A])
        # x = vutils.make_grid(C, nrow=4)
        # plt.imshow(np.transpose(x, (1, 2, 0)))

        if i_batch > 4:
            break

        AB_batch = np.concatenate([A, B], 3)
        for i in range(len(AB_batch)):
            im_AB = (AB_batch[i].transpose([1, 2, 0])*255).astype(np.uint8)
            np.clip(im_AB, 0, 255, im_AB)
            path_AB = output_dir + '/{}/{}.png'.format('val' if counter > train_split else 'train', counter)
            io.imsave(path_AB, im_AB)
            plt.imshow(im_AB)
            plt.show()
            counter += 1
        if counter > nb_images:
            break
