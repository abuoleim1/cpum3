import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from const import device
from tqdm import tqdm

class FER2013Dataset(Dataset):
    def __init__(self, csv_file, usage='Training', transform=None):
        """
        Args:
            csv_file (str): Path to the csv file with annotations.
            usage (str): Which usage partition to use ('Training', 'PublicTest', or 'PrivateTest').
            transform (callable, optional): Optional transform to be applied on an image.
            device (torch.device, optional): Device to load data to (e.g., torch.device('cuda')).
        """
        self.transform = transform
        self.device = device
        self.usage = usage

        # Load and filter data
        data = pd.read_csv(csv_file)
        data = data[data['Usage'] == usage].reset_index(drop=True)

        self.images = []
        self.labels = []

        for _, row in tqdm(data.iterrows(), desc="Transforming data", total=len(data)):
            label = int(row['emotion'])
            pixels = np.array(row['pixels'].split(), dtype='float32').reshape(48, 48)
            image = Image.fromarray(np.uint8(pixels))
            if self.transform:
                image = self.transform(image)
            else:
                # Convert to tensor if no transform is provided
                image = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0) / 255.0  # default normalization
            self.images.append(image.to(self.device))
            self.labels.append(torch.tensor(label, dtype=torch.long).to(self.device))
        
        self.get_distribution()
        
    def get_distribution(self):
        label_counts = {}
        for label in self.labels:
            label_int = label.item()
            if label_int not in label_counts:
                label_counts[label_int] = 0
            label_counts[label_int] += 1

        print(f"Dataset '{self.usage}' class distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"Class {label}: {count} samples")
        print(f"Total: {len(self.labels)} samples")
        
        return label_counts
        
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
