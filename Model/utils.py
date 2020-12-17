from PIL import Image
from torch.utils.data import Dataset, DataLoader
from . import dense_transforms


class DetectionDataset(Dataset):
    def __init__(self, dataset_path, transform=dense_transforms.ToTensor(), min_size=20):
        from glob import glob
        from os import path
        self.files = []
        for im_f in glob(path.join(dataset_path, '*_im.jpg')):
            self.files.append(im_f.replace('_im.jpg', ''))
        self.transform = transform
        self.min_size = min_size

    def _filter(self, boxes):
        if len(boxes) == 0:
            return boxes
        return boxes[abs(boxes[:, 3] - boxes[:, 1]) * abs(boxes[:, 2] - boxes[:, 0]) >= self.min_size]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        import numpy as np
        b = self.files[idx]
        im = Image.open(b + '_im.jpg')
        nfo = np.load(b + '_boxes.npz')
        data = im, self._filter(nfo['karts']), self._filter(nfo['bombs']), self._filter(nfo['pickup'])
        if self.transform is not None:
            data = self.transform(*data)
        return data


def load_detection_data(dataset_path, num_workers=4, batch_size=32, **kwargs):
    dataset = DetectionDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


if __name__ == '__main__':
    dataset = DetectionDataset('Data/dense_data/train')
    import torchvision.transforms.functional as F
    from pylab import show, subplots
    import matplotlib.patches as patches
    import numpy as np

    fig, axs = subplots(1, 2)
    #todo
    #vizualization of data
