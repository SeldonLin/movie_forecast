import os
import torch
import ast
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from encoder_decoder import TextEmbedder


ImageFile.LOAD_TRUNCATED_IMAGES = True


class DatasetPrepare():
    def __init__(self, data, comment, introduction,pic_root, photos_root,embedding_dim, gpu_num):
        self.data = data
        self.comment = comment
        self.introduction = introduction
        self.embedding_dim = embedding_dim
        self.pic_root = pic_root
        self.photos_root = photos_root
        self.introduction_encoder = TextEmbedder(embedding_dim, gpu_num)
        self.gpu_num = gpu_num

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx, :]

    def preprocess_data(self):
        data = self.data
        comments_text_encoding = self.comment.detach()
        introduction_encoding = self.introduction.detach()
        embedding_dim = self.embedding_dim
        
        # get comment 

        # Read the descriptions and the images
        image_features = []
        img_transforms = Compose(
            [Resize((256, 64)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        for (idx, row) in tqdm(data.iterrows(), total=len(data), ascii=True):
            id, name, director, actor, introduction = row['id'], row['name'], row['director'], row['actor'], row['introduction']

            # Read images
            pic_path =  f"{id}.png"
            photo_0_path = f"{id}_0.png"
            photo_1_path = f"{id}_1.png"
            photo_2_path = f"{id}_2.png"

            pic = Image.open(os.path.join(self.pic_root, pic_path)).convert('RGB')
            photo_0 = Image.open(os.path.join(self.photos_root, photo_0_path)).convert('RGB')
            photo_1 = Image.open(os.path.join(self.photos_root, photo_1_path)).convert('RGB')
            photo_2 = Image.open(os.path.join(self.photos_root, photo_2_path)).convert('RGB')
            image_features_tmp = []
            image_features_tmp.append(img_transforms(pic))
            image_features_tmp.append(img_transforms(photo_0))
            image_features_tmp.append(img_transforms(photo_1))
            image_features_tmp.append(img_transforms(photo_2))
            images = torch.stack(image_features_tmp)
            images = images.reshape(3,256,-1)
            image_features.append(images)

        # Create TensorDataset
        box_office = torch.FloatTensor(data.iloc[:,-24:-4].values)
        # introduction_encoding = self.introduction_encoder(movies_description).to('cpu')
        images = torch.stack(image_features)
        temporal_features = torch.FloatTensor(data[['day','week','month','year']].values)
        # box_office:(data_len,20) (706,20);   introduction_encoding:(data_len,embedding_dim) (706,32);
        # images:(data_len,RGB,img_size[0],img_size[1]);   temporal_features: (data_len, 4)(706, 4)
        # comments_encoding(data_len,comment_len,embedding_dim) (706,100,32);
        return TensorDataset(box_office, introduction_encoding, comments_text_encoding, images, temporal_features)

    def get_loader(self, batch_size, train=True):
        print('Starting dataset creation process...')
        data = self.preprocess_data()
        data_loader = None
        if train:
            data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)
        else:
            data_loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=4)
        print('Done.')

        return data_loader