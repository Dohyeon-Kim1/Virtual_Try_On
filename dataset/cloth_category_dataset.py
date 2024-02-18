import os
import random
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


DATA_PATH = "/mnt/sdc/dhkim/ClothCategory"


class ClothCategoryDataset(Dataset):
    def __init__(self, img_size=(128,96)):
        self.img_path = []
        self.label = []
        self.label_mapping = {}
        self.sample_per_category = min([len(os.listdir(f"{DATA_PATH}/{category}")) for category in os.listdir(DATA_PATH)])
        for label, category in enumerate(os.listdir(DATA_PATH)):
            self.label_mapping[label] = category
            paths = os.listdir(f"{DATA_PATH}/{category}")
            random.shuffle(paths)
            paths = paths[:self.sample_per_category]
            for path in paths:
                self.img_path.append(f"{DATA_PATH}/{category}/{path}")
                self.label.append(label)

        self.img_size = img_size
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                                             transforms.Resize(img_size)])
    
    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, i):
        return self.transform(Image.open(self.img_path[i]).convert("RGB")), self.label[i]