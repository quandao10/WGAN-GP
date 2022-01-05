import os
from torch.utils.data import Dataset
from PIL import Image


class FaceGan(Dataset):
    def __init__(self, root_dir, transform=None):
        super(FaceGan, self).__init__()
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        image_name = "{:0>6}.jpg".format(index + 1)
        image = Image.open(os.path.join(self.root_dir, image_name))
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(os.listdir(self.root_dir))
