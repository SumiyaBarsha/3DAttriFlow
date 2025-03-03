import torch
import numpy as np
import torch.utils.data as data
import h5py
import os
import torchvision.transforms as transforms
from PIL import Image

NUM_VIEWS = 24  # Define the number of views per object
IMAGE_DIR = "E:\data\ShapeNetV1Renderings"  # Path where multi-view images are stored



class MVP_CP(data.Dataset):
    def __init__(self, prefix="train", image_dir=IMAGE_DIR, transform=None):
        if prefix == "train":
            self.file_path = 'E:\\3DAttriFlow\\MVP_Train_CP.h5'
        elif prefix == "test":
            self.file_path = 'E:\\3DAttriFlow\\MVP_Test_CP.h5'
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")

        self.prefix = prefix
        self.image_dir = image_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),  # Resize for ResNet input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        input_file = h5py.File(self.file_path, 'r')

        self.input_data = np.array(input_file['incomplete_pcds'][()])
        self.labels = np.array(input_file['labels'][()])
        self.categorys = self.labels

        # Label one-hot encoding
        num_classes = 16
        self.labels = np.eye(num_classes)[self.labels]

        self.gt_data = np.array(input_file['complete_pcds'][()])
        
        input_file.close()
        self.len = self.input_data.shape[0]

    def load_image(self, image_path):
        """ Load and transform an image """
        if os.path.exists(image_path):
            img = Image.open(image_path).convert('RGB')
            return self.transform(img)
        else:
            return torch.zeros(3, 224, 224)  # Return black image if not found

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = torch.from_numpy(self.input_data[index])  #partial point cloud

        if self.prefix != "test":
            complete = torch.from_numpy(self.gt_data[index // 26]) #use object index for GT
            labels = torch.tensor(self.labels[index], dtype=torch.float32)

            # Load multi-view images for the corresponding object
            images = []
            obj_id = index//26  # Determine object ID from sample index (26 samples per object)
            for view_id in range(NUM_VIEWS):
                image_path = f"{self.image_dir}/{obj_id}_view{view_id}.jpg"
                images.append(self.load_image(image_path))

            images = torch.stack(images)  # Stack multi-view images (Shape: [NUM_VIEWS24, 3, 224, 224])

            return labels, partial, complete, images  # Include multi-view images
        else:
            complete = torch.from_numpy(self.gt_data[index // 26])
            labels = torch.tensor(self.labels[index], dtype=torch.float32)
            categorys = torch.tensor(self.categorys[index])

            return categorys, labels, partial, complete
