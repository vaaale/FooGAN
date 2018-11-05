from torchvision import transforms

from dataset import get_folder_loader
from imageutils import distort
import numpy as np

dataroot = 'Y:/Dataset/B_Gray'

tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.Lambda(distort),
])

if __name__ == '__main__':
    dataloader = get_folder_loader(dataroot=dataroot, transform=tf, batch_size=4096, shuffle=False, num_workers=1)

    pop_mean = []
    pop_std0 = []
    pop_std1 = []
    for i, (data, _) in enumerate(dataloader, 0):
        # shape (batch_size, 3, height, width)
        numpy_image = data['B'].numpy()

        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
        batch_std0 = np.std(numpy_image, axis=(0, 2, 3))
        batch_std1 = np.std(numpy_image, axis=(0, 2, 3), ddof=1)

        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)
        pop_std1.append(batch_std1)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std0 = np.array(pop_std0).mean(axis=0)
    pop_std1 = np.array(pop_std1).mean(axis=0)

    print('pop_mean: {}'.format(pop_mean))
    print('pop_std0: {}'.format(pop_std0))
    print('pop_std1: {}'.format(pop_std1))

# Train A
# pop_mean: [0.52646667 0.52646667 0.52646667]
# pop_std0: [0.21443187 0.21443187 0.21443187]
# pop_std1: [0.21443187 0.21443187 0.21443187]

# Train B
# pop_mean: [0.5854925 0.5845946 0.5835283]
# pop_std0: [0.28469142 0.28471425 0.28451768]
# pop_std1: [0.28469145 0.28471425 0.28451774]


# Val
# pop_mean: [0.5846711 0.5846711 0.5846711]
# pop_std0: [0.2865914 0.2865914 0.2865914]
# pop_std1: [0.28659144 0.28659144 0.28659144]
