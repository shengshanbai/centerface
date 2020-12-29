import os
import argparse
from dataset import WFaceDB
from dataset import wface_collate_fn
from dataset import cpu_collate_fn
import torch
from config import g_config
import models
from centerface_trainer import CenterFaceTrainer
import tqdm
import logging
import torch.multiprocessing as mp

def main(args):
    logging.basicConfig(filename=args.log_file, level=logging.DEBUG)
    train_dataset = WFaceDB(args.train_db,True)
    val_dataset = WFaceDB(args.val_db,False)
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size,
                                               True, num_workers=args.num_workers,collate_fn=cpu_collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, args.batch_size,
                                              num_workers=args.num_workers,collate_fn=cpu_collate_fn)
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_file = os.path.join(model_dir, 'centerface.pth.tar')
    epoch_size = len(train_loader)
    print('Number of batches per epoch:{}'.format(epoch_size))
    model = models.CenterFaceNet(args.net_ratio)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-7)
    trainer = CenterFaceTrainer(model, optimizer)
    trainer.set_device(g_config.device)
    start_epoch = 0
    if os.path.exists(model_file):
        start_epoch = trainer.load_state(args.model_dir)
    max_epoch = args.max_epoch
    for epoch in tqdm.tqdm(range(start_epoch, max_epoch), total=max_epoch - start_epoch, desc='train'):
        train_loss = trainer.train(epoch, train_loader)
        trainer.save_state(args.model_dir, epoch)
        val_loss = trainer.test(epoch, val_loader)
        logging.info("epoch:{} train loss:{} val loss:{}".format(epoch,train_loss,val_loss))
    train_dataset.close()
    val_dataset.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--train_db', default=r'C:\Users\sheng\Downloads\data\WIDER\train_lmdb', help='the train lmdb dir')
    parser.add_argument('--val_db', default=r'C:\Users\sheng\Downloads\data\WIDER\val_lmdb', help='the val lmdb dir')
    parser.add_argument('--num_workers', default=0, type=int, help='num of data loader thread')
    parser.add_argument('--batch_size', default=1, type=int, help='train batch size')
    parser.add_argument('--model_dir', default=r'./models', help='the model save dir')
    parser.add_argument('--log_file', default='./log.txt', help='the log file')
    parser.add_argument('--max_epoch', type=int, default=80)
    parser.add_argument('--net_ratio', default=1.0, type=float, help='the network width ratio')
    args=parser.parse_args()
    main(args)