import torch
import torch.nn as nn
from utils import AverageMeter
import os
import tqdm


class ModelWithLoss(nn.Module):
    def __init__(self, model, loss):
        super().__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        outputs = self.model(batch[0])
        loss = self.loss(outputs, batch)
        return outputs, loss


class BaseTrainer:
    def __init__(self, model, optimizer,scheduler=None):
        super().__init__()
        self.optimizer = optimizer
        self.model_with_loss = ModelWithLoss(model, self._get_losses())
        self.scheduler=scheduler

    def _get_losses(self):
        raise NotImplementedError()

    def set_device(self, device):
        self.device = device
        self.model_with_loss = self.model_with_loss.to(device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)

    def test(self, epoch, data_loader):
        return self.run_epoch('test', epoch, data_loader)

    def run_step(self,phase,batch):
        outputs, loss = self.model_with_loss(batch)
        if phase == 'train':
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler:
                self.scheduler.step()
        return outputs,loss

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            model_with_loss.eval()
            torch.cuda.empty_cache()
        mean_loss = AverageMeter()
        self.optimizer.zero_grad()
        for iter_id, batch in tqdm.tqdm(enumerate(data_loader), desc='epoch {}'.format(epoch), total=len(data_loader)):
            for i in range(len(batch)):
                #batch[i] = batch[i].to(device=self.device, non_blocking=True)
                batch[i] = torch.from_numpy(batch[i]).to(device=self.device,non_blocking=True)
            outputs,loss=self.run_step(phase,batch)
            mean_loss.update(loss.cpu().item())
        return mean_loss.avg

    def save_state(self, model_dir, epoch):
        save_path = os.path.join(model_dir, 'centerface.pth.tar')
        state_dict = {'centerface': self.model_with_loss.model.state_dict(),
                      'epoch': epoch,
                      'optimizer': self.optimizer.state_dict()}
        torch.save(state_dict, save_path)

    def load_state(self, model_dir):
        file_path = os.path.join(model_dir, 'centerface.pth.tar')
        param_dict = torch.load(file_path)
        self.model_with_loss.model.load_state_dict(param_dict['centerface'])
        #不保存学习率，方便手动调整
        lr=self.optimizer.param_groups[0]["lr"]
        self.optimizer.load_state_dict(param_dict['optimizer'])
        self.optimizer.param_groups[0]["lr"]=lr
        return param_dict['epoch']
