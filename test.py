import scipy.io as sio
import torch.utils.data as data
import os
import lmdb
import numpy as np
import cv2
import models
import torch
from config import g_config

class ValImageDataset(data.Dataset):
    def __init__(self, lmdb_dir):
        buf_size = os.path.getsize(os.path.join(lmdb_dir, 'data.mdb'))
        self.db_env = lmdb.open(lmdb_dir, buf_size)
        self.txn = self.db_env.begin()
        self.size = np.frombuffer(self.txn.get("len".encode()), dtype=np.int32)[0]

    def __getitem__(self, index):
        image_content = self.txn.get("image_{}".format(index).encode())
        image_content = np.asarray(bytearray(image_content), dtype="uint8")
        image = cv2.imdecode(image_content, cv2.IMREAD_COLOR)
        image = image.astype(np.float32) / 255.0
        fname=self.txn.get("fname_{}".format(index).encode()).decode()
        return fname,image

    def __len__(self):
        return self.size


if __name__=='__main__':
    model = models.CenterFaceNet()
    model.to(g_config.device)
    model.eval()
    param_dict = torch.load(r'./models/centerface.pth.tar')
    model.load_state_dict(param_dict['centerface'])
    val_db=ValImageDataset(r'C:\Users\sheng\Downloads\data\WIDER\val_lmdb')
    for item in val_db:
        fname,image=item