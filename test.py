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

def preprocessImage(image):
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                         dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                        dtype=np.float32).reshape(1, 1, 3)
    height,width=image.shape[0:2]
    pad_right=(32-width%32)%32
    pad_bottom=(32-height%32)%32
    if pad_right!=0 or pad_bottom!=0:
        image=cv2.copyMakeBorder(image,0,pad_bottom,0,pad_right,cv2.BORDER_CONSTANT)
    return (image-mean)/std

def nms(bboxes):
    sorted, indices=torch.sort(bboxes[:,4],descending=True)
    bboxes=bboxes[indices,:]
    added_ids=[]
    merged_boxes=[]
    box_sizes=(bboxes[:,2]-bboxes[:,0])*(bboxes[:,3]-bboxes[:,1])
    for i in range(bboxes.shape[0]):
        if i not in added_ids:
            cur_box=bboxes[i]
            added_ids.append(i)
            merged_boxes.append(cur_box.detach().cpu().numpy())
            cur_size=box_sizes[i]
            left_size=box_sizes[i+1:]
            inner_x0=bboxes[i+1:,0].clone()
            inner_x0[inner_x0<cur_box[0]]=cur_box[0]
            inner_y0=bboxes[i+1:,1].clone()
            inner_y0[inner_y0<cur_box[1]]=cur_box[1]
            inner_x1=bboxes[i+1:,2].clone()
            inner_x1[inner_x1>cur_box[2]]=cur_box[2]
            inner_y1=bboxes[i+1:,3].clone()
            inner_y1[inner_y1>cur_box[3]]=cur_box[3]
            inner_sizes=(inner_y1-inner_y0)*(inner_x1-inner_x0)
            ratio=inner_sizes/(left_size+cur_size-inner_sizes)
            tomerge_ids=torch.where(ratio>0.3)[0]
            tomerge_ids+=(i+1)
            added_ids.extend(tomerge_ids.cpu().tolist())
    if len(merged_boxes)>0:
        return np.vstack(merged_boxes)
    return None

def parseOutput(outs):
    heat_map, center_off, wh, landmarks=outs
    pos_index=torch.where(heat_map>0.5)
    cy_int=pos_index[2]
    cx_int=pos_index[3]
    cx_off=center_off[pos_index]
    cy_off=center_off[pos_index[0],pos_index[1]+1,pos_index[2],pos_index[3]]
    cx=cx_int+cx_off
    cy=cy_int+cy_off
    w=torch.exp(wh[pos_index])
    h=torch.exp(wh[pos_index[0],pos_index[1]+1,pos_index[2],pos_index[3]])
    bboxes=torch.zeros((w.shape[0],5))
    bboxes[:,0]=cx-w/2
    bboxes[:,1]=cy-h/2
    bboxes[:,2]=cx+w/2
    bboxes[:,3]=cy+h/2
    bboxes[:,4]=heat_map[pos_index]
    return nms(bboxes)

def limitBoxesRange(bboxes,x_max,y_max):
    bboxes[:,0:4]*=4
    bboxes[:,[0,2]]=np.clip(bboxes[:,[0,2]],0,x_max)
    bboxes[:,[1,3]]=np.clip(bboxes[:,[1,3]],0,y_max)
    return bboxes

if __name__=='__main__':
    model = models.CenterFaceNet()
    model.to(g_config.device)
    model.eval()
    param_dict = torch.load(r'./models/centerface.pth.tar')
    model.load_state_dict(param_dict['centerface'])
    val_db=ValImageDataset(r'C:\Users\sheng\Downloads\data\WIDER\val_lmdb')
    for item in val_db:
        fname,image=item
        fimage=preprocessImage(image)
        fimage=np.transpose(fimage,[2,0,1])
        cuda_image=torch.from_numpy(fimage[np.newaxis,:,:,:]).to(g_config.device)
        outs=model(cuda_image)
        bboxes=parseOutput(outs)
        if bboxes is not None:
            bboxes=limitBoxesRange(bboxes,image.shape[1]-1,image.shape[0]-1)
            for i in range(bboxes.shape[0]):
                box=bboxes[i]
                cv2.rectangle(image,(box[0],box[1]),(box[2],box[3]),(0,255,255),thickness=1)
        cv2.imshow("image",image)
        cv2.waitKey()
        cv2.destroyWindow("image")
