from base_trainer import BaseTrainer
from loss import CenterLoss

class CenterFaceTrainer(BaseTrainer):
    def __init__(self, model, optimizer,scheduler=None):
        super().__init__(model, optimizer,scheduler)

    def _get_losses(self):
        return CenterLoss()