import lmdb
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
import cv2
import io
import random
import math
import torch
from config import g_config


def Data_anchor_sample(image, face_boxes, face_keys=None):
    # 增加小人脸的比例
    maxSize = 12000
    infDistance = 9999999

    height, width, _ = image.shape
    random_counter = 0
    box_area = face_boxes[:, 2] * face_boxes[:, 3]
    rand_idx = random.randint(0, len(box_area) - 1)  # 随机选择一个人脸
    if box_area[rand_idx]<=0: #有的人脸大小被标记为0，可以跳过这些数据
        if height*width  > maxSize * maxSize:
            ratio = (maxSize * maxSize / (height * width)) ** 0.5
            interp_method = cv2.INTER_LINEAR
            image = cv2.resize(image, None, None, fx=ratio, fy=ratio, interpolation=interp_method)
            face_boxes = face_boxes * ratio
            if face_keys is not None:
                face_keys[:, :, 0:2] *= ratio
        return image, face_boxes, face_keys
    rand_Side = box_area[rand_idx] ** 0.5  # 等效正方形边长

    anchors = [16, 32, 48, 64, 96, 128, 256, 512]
    distance = infDistance
    anchor_idx = 5
    for i, anchor in enumerate(anchors):
        if abs(anchor - rand_Side) < distance:
            distance = abs(anchor - rand_Side)  # 选择最接近的anchors
            anchor_idx = i

    target_anchor = random.choice(anchors[0:min(anchor_idx + 1, 5)])  # 随机选择一个相对较小的anchor，向下
    ratio = float(target_anchor) / rand_Side  # 缩放的尺度
    ratio = ratio * (2 ** random.uniform(-1, 1))  # [ratio/2, 2ratio]的均匀分布

    if height * ratio * width * ratio > maxSize * maxSize:
        ratio = (maxSize * maxSize / (height * width)) ** 0.5

    interp_method = cv2.INTER_LINEAR
    image = cv2.resize(image, None, None, fx=ratio, fy=ratio, interpolation=interp_method)

    face_boxes = face_boxes * ratio
    if face_keys is not None:
        face_keys[:, :, 0:2] *= ratio
    return image, face_boxes, face_keys


def get_border(border, size):
    i = 1
    while size - border // i <= border // i:  # 必须剩下部分图像
        i *= 2
    return border // i


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return a + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_center_scale(image, is_train):
    c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
    s = max(image.shape[0], image.shape[1]) * 1.0
    rot = 0
    flipped = False
    if is_train:
        _border = s * np.random.choice([0.1, 0.2, 0.25])
        w_border = get_border(_border, image.shape[1])
        h_border = get_border(_border, image.shape[0])
        c[0] = np.random.randint(low=w_border, high=image.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=image.shape[0] - h_border)
        if np.random.random() < 0.5:
            flipped = True
    return c, s, rot, flipped


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3,))
    image += np.dot(eigvec, eigval * alpha)


def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2


def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])


def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha


def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)


def color_aug(data_rng, image, eig_val, eig_vec):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.3)
    lighting_(data_rng, image, 0.1, eig_val, eig_vec)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1  # 直径
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]  # 对那个区域进行赋值
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # 取重叠的最大值
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def affine_transform(pts, aff_mat):
    origin_shape = pts.shape
    trans_pts = pts.reshape(-1, 2).T
    hom_pts = np.ones((trans_pts.shape[0] + 1, trans_pts.shape[1]), trans_pts.dtype)
    hom_pts[0:2, :] = trans_pts
    trans_out = np.dot(aff_mat, hom_pts)
    return trans_out.T.reshape(origin_shape)


def gaussian_radius(det_size, min_overlap=0.7):
    # 算法的思想参考https://zhuanlan.zhihu.com/p/96856635
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


class WFaceDB(data.Dataset):
    def __init__(self, lmdb_dir, is_train, shuffle=True):
        buf_size = os.path.getsize(os.path.join(lmdb_dir, 'data.mdb'))
        self.db_env = lmdb.open(lmdb_dir,buf_size)
        self.is_train = is_train
        self.txn = self.db_env.begin()
        self.size = np.frombuffer(self.txn.get("len".encode()), dtype=np.int32)[0]
        self.eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                dtype=np.float32)
        self.eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        self.mean = np.array([0.40789654, 0.44719302, 0.47026115],
                             dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.28863828, 0.27408164, 0.27809835],
                            dtype=np.float32).reshape(1, 1, 3)
        self.data_rng = np.random.RandomState(123)

    def __getitem__(self, index):
        image_content = self.txn.get("image_{}".format(index).encode())
        image_content = np.asarray(bytearray(image_content), dtype="uint8")
        image = cv2.imdecode(image_content, cv2.IMREAD_COLOR)
        image = image.astype(np.float32) / 255.0
        has_face = np.frombuffer(self.txn.get("has_face_{}".format(index).encode()), dtype=np.bool)[0]
        if has_face:
            face_boxes = np.frombuffer(self.txn.get("face_boxes_{}".format(index).encode()), dtype=np.int32)
            face_boxes = face_boxes.reshape(-1, 4).copy()
            face_keys = np.ones((face_boxes.shape[0], 5, 3), dtype=np.float32) * -1
            if self.is_train:
                face_keys = np.frombuffer(self.txn.get("face_keys_{}".format(index).encode()), dtype=np.float32)
                face_keys = face_keys.reshape(-1, 5, 3).copy()
                blurs = np.frombuffer(self.txn.get("blurs_{}".format(index).encode()), dtype=np.float32).copy()
            image, face_boxes, face_keys = Data_anchor_sample(image, face_boxes, face_keys)
        height, width, _ = image.shape
        c, s, rot, flipped = get_center_scale(image, self.is_train)
        if flipped:
            image = image[:, ::-1, :]
            c[0] = width - c[0] - 1
        trans_input = get_affine_transform(
            c, s, rot, [g_config.input_res, g_config.input_res])
        input_image = cv2.warpAffine(image, trans_input,
                                     (g_config.input_res, g_config.input_res),
                                     flags=cv2.INTER_LINEAR)
        # 随机图片增强
        if self.is_train:
            color_aug(self.data_rng, input_image, self.eig_val, self.eig_vec)
        input_image = (input_image - self.mean) / self.std
        input_image = input_image.transpose(2, 0, 1)

        output_res = g_config.output_res
        trans_output = get_affine_transform(c, s, rot, [output_res, output_res])
        num_faces = 1
        if has_face:
            num_faces = len(face_boxes)
        heat_map = np.zeros((1, output_res, output_res), dtype=np.float32)
        wh_gt = np.zeros((num_faces, 2), dtype=np.float32)
        face_index = np.zeros((num_faces), dtype=np.int32)
        face_off = np.zeros((num_faces, 2), dtype=np.float32)
        face_mask = np.zeros((num_faces), dtype=np.bool)
        face_key_off = np.zeros((num_faces, 5 * 2), dtype=np.float32)
        face_key_mask = np.zeros((num_faces, 5 * 2), dtype=np.bool)
        if has_face:
            boxes = face_boxes.copy()
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
            if flipped:
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]] - 1
                face_keys[:, :, 0] = width - face_keys[:, :, 0] - 1
                face_keys[:, [0, 1, 3, 4], :] = face_keys[:, [1, 0, 4, 3], :]
            boxes[:, 0:2] = affine_transform(boxes[:, 0:2], trans_output)
            boxes[:, 2:4] = affine_transform(boxes[:, 2:4], trans_output)
            face_keys[:, :, 0:2] = affine_transform(face_keys[:, :, 0:2], trans_output)
            boxes = np.clip(boxes, 0, output_res - 1)
            boxes_w = boxes[:, 2] - boxes[:, 0]
            boxes_h = boxes[:, 3] - boxes[:, 1]
            for i in range(num_faces):
                w = boxes_w[i]
                h = boxes_h[i]
                if w > 0 and h > 0:
                    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius = max(0, int(radius))
                    wh_gt[i] = np.log(w), np.log(h)
                    # 人脸中心点
                    face_center = np.array([(boxes[i, 0] + boxes[i, 2]) / 2, (boxes[i, 1] + boxes[i, 3]) / 2],
                                           dtype=np.float32)
                    face_center_int = face_center.astype(np.int32)
                    face_index[i] = face_center_int[1]*output_res+face_center_int[0]  # 人脸bbox在1/4特征图中的索引
                    face_off[i] = face_center - face_center_int  # 人脸box中心点整数化的偏差
                    face_mask[i] = 1
                    if face_keys[i, 0, 2] >= 0 and w * h > 2:  # 太小的人脸忽略
                        for j in range(5):
                            if face_keys[i, j, 0] >= 0 and face_keys[i, j, 0] < output_res and \
                                    face_keys[i, j, 1] >= 0 and face_keys[i, j, 1] < output_res:
                                face_key_off[i, 2 * j] = (face_keys[i, j, 0] - face_center_int[0]) / w
                                face_key_off[i, 2 * j + 1] = (face_keys[i, j, 1] - face_center_int[1]) / h
                                face_key_mask[i, 2 * j:2 * j + 2] = 1
                    draw_gaussian(heat_map[0], face_center_int, radius)
        return (input_image, heat_map, wh_gt, face_index, face_off, face_mask, face_key_off, face_key_mask)

    def __len__(self):
        return self.size

def wface_collate_fn(batch):
    out_list = []
    data_class = len(batch[0])
    max_face=-1
    for i in range(data_class):
        if i <= 1:  # 图像数据
            c_list=[torch.from_numpy(row[i]).to(g_config.device) for row in batch]
            image_batch=torch.stack(c_list)
            out_list.append(image_batch)
        else: #其它数据
            if max_face == -1:
                max_face=max([row[i].shape[0] for row in batch])
            c_list=[]
            for k in range(len(batch)):
                data=torch.from_numpy(batch[k][i]).to(g_config.device)
                or_shape=data.shape
                temp_shape=list(or_shape)
                temp_shape[0]=max_face
                temp_data=torch.zeros(temp_shape,dtype=data.dtype,device=g_config.device)
                temp_data[0:or_shape[0]]=data
                c_list.append(temp_data)
            data_batch=torch.stack(c_list)
            out_list.append(data_batch)
    return out_list

def cpu_collate_fn(batch):
    out_list = []
    data_class = len(batch[0])
    max_face = -1
    for i in range(data_class):
        if i <= 1:  # 图像数据
            c_list = [row[i] for row in batch]
            image_batch = np.stack(c_list)
            out_list.append(image_batch)
        else:  # 其它数据
            if max_face == -1:
                max_face = max([row[i].shape[0] for row in batch])
            c_list = []
            for k in range(len(batch)):
                data = batch[k][i]
                or_shape = data.shape
                temp_shape = list(or_shape)
                temp_shape[0] = max_face
                temp_data = np.zeros(temp_shape, dtype=data.dtype)
                temp_data[0:or_shape[0]] = data
                c_list.append(temp_data)
            data_batch = np.stack(c_list)
            out_list.append(data_batch)
    return out_list


if __name__ == "__main__":
    reader = WFaceDB(r'C:\Users\sheng\Downloads\data\WIDER\train_lmdb', True)
    for i in range(len(reader)):
        data_item = reader[i]
        image = data_item[0]
        image = image.transpose(1, 2, 0)
        image = image * reader.std + reader.mean
        heat_map = data_item[1]
        heat_map = heat_map.transpose(1, 2, 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        heat_map = cv2.resize(heat_map, (image.shape[1], image.shape[0]))
        heat_map = cv2.cvtColor(heat_map, cv2.COLOR_GRAY2RGB)
        merge_image = image + heat_map
        face_mask = data_item[5]
        for j in range(len(face_mask)):
            if face_mask[j] > 0:
                face_index = data_item[3][j]
                center_int=[face_index%g_config.output_res,face_index//g_config.output_res]
                wh = np.exp(data_item[2][j])
                face_off = data_item[4][j]
                face_center = center_int + face_off
                pt1 = (face_center - wh / 2) * 4
                pt2 = (face_center + wh / 2) * 4
                cv2.rectangle(merge_image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 255, 0), 2)
                face_key_off = data_item[6]
                face_key_mask = data_item[7]
                for k in range(4):
                    if face_key_mask[j, 2 * k] > 0:
                        face_key_x = face_key_off[j, 2 * k] * wh[0] + center_int[0]
                        face_key_y = face_key_off[j, 2 * k + 1] * wh[1] + center_int[1]
                        cv2.circle(merge_image, (int(face_key_x * 4), int(face_key_y * 4)), 1, (255, 255, 0))
        plt.imshow(merge_image)
        plt.show()
