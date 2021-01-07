# This is a sample Python script.
import os
import argparse
import numpy as np
import lmdb
import cv2


def parse_desc_file(desc_file):
    rst = []
    with open(desc_file, 'r') as desc_f:
        data_item = {}
        for line in desc_f:
            if line.startswith("#"):
                if "image_fname" in data_item:
                    rst.append(data_item)
                    data_item = {}
                image_fname = line[1:].strip()
                data_item["image_fname"] = image_fname
            else:
                parts = line.strip().split()
                if "face_boxes" not in data_item:
                    data_item["face_boxes"] = []
                data_item["face_boxes"].append(np.array([int(x) for x in parts[0:4]], dtype=np.int32))
                if len(parts) > 4:
                    if "face_keys" not in data_item:
                        data_item["face_keys"] = []
                    data_item["face_keys"].append(
                        np.array([float(x) for x in parts[4:19]], dtype=np.float32).reshape(5, 3))
                    if "blur" not in data_item:
                        data_item["blur"] = []
                    data_item["blur"].append(float(parts[19]))
        # the last item
        rst.append(data_item)
    return rst


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
def main(args):
    train_dir = os.path.join(args.wider_dir, "train")
    train_desc_file = os.path.join(train_dir, "label.txt")
    train_image_dir = os.path.join(train_dir, "images")
    train_desc = parse_desc_file(train_desc_file)
    train_lmdb_path = os.path.join(args.wider_dir, "train_lmdb")
    train_lmdb = lmdb.open(train_lmdb_path, map_size=1.5 * 1024 * 1024 * 1024)
    for index, item in enumerate(train_desc):
        image_file = os.path.join(train_image_dir, item["image_fname"])
        with open(image_file, 'rb') as image_f:
            image_content = image_f.read()
        if "face_boxes" in item:
            has_face = True
            face_boxes = np.stack(item["face_boxes"], axis=0)
            face_keys = np.stack(item["face_keys"], axis=0)
            face_blurs = np.array(item["blur"], dtype=np.float32)
            with train_lmdb.begin(write=True) as txn:
                txn.put("fname_{}".format(index).encode(),item["image_fname"].encode())
                txn.put("image_{}".format(index).encode(), image_content)
                txn.put("has_face_{}".format(index).encode(), np.array(has_face, dtype=np.bool))
                txn.put("blurs_{}".format(index).encode(), face_blurs)
                txn.put("face_boxes_{}".format(index).encode(), face_boxes)
                txn.put("face_keys_{}".format(index).encode(), face_keys)
        else:
            has_face = False
            with train_lmdb.begin(write=True) as txn:
                txn.put("fname_{}".format(index).encode(), item["image_fname"].encode())
                txn.put("image_{}".format(index).encode(), image_content)
                txn.put("has_face_{}".format(index).encode(), np.array(has_face, dtype=np.bool))
        if index % 100 == 0:
            print("proccess train {}".format(index))
    with train_lmdb.begin(write=True) as txn:
        txn.put("len".encode(), np.array(len(train_desc), dtype=np.int32))

    val_dir = os.path.join(args.wider_dir, "val")
    val_desc_file = os.path.join(val_dir, "label.txt")
    val_image_dir = os.path.join(val_dir, "images")
    val_desc = parse_desc_file(val_desc_file)
    val_lmdb_path = os.path.join(args.wider_dir, "val_lmdb")
    val_lmdb = lmdb.open(val_lmdb_path, map_size=360 * 1024 * 1024)
    for index, item in enumerate(val_desc):
        image_file = os.path.join(val_image_dir, item["image_fname"])
        with open(image_file, 'rb') as image_f:
            image_content = image_f.read()
        if "face_boxes" in item:
            has_face = True
            face_boxes = np.stack(item["face_boxes"], axis=0)
            with val_lmdb.begin(write=True) as txn:
                txn.put("fname_{}".format(index).encode(), item["image_fname"].encode())
                txn.put("image_{}".format(index).encode(), image_content)
                txn.put("has_face_{}".format(index).encode(), np.array(has_face, dtype=np.bool))
                txn.put("face_boxes_{}".format(index).encode(), face_boxes)
        else:
            has_face = False
            with val_lmdb.begin(write=True) as txn:
                txn.put("fname_{}".format(index).encode(), item["image_fname"].encode())
                txn.put("image_{}".format(index).encode(), image_content)
                txn.put("has_face_{}".format(index).encode(), np.array(has_face, dtype=np.bool))
        if index % 100 == 0:
            print("proccess val {}".format(index))
    with val_lmdb.begin(write=True) as txn:
        txn.put("len".encode(), np.array(len(val_desc), dtype=np.int32))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser("cmd parser!")
    arg_parser.add_argument("--wider_dir", default=r'C:\Users\sheng\Downloads\data\WIDER', help='the wider faces dir')
    args = arg_parser.parse_args()
    main(args)
