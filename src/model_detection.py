import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.model_retrieval import Model as SBIR_Model
from src.networks import WSDDN
from experiments.options import opts

def freeze_model(m):
    m.requires_grad_(False)

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.opts = opts
        self.sbir_model = SBIR_Model().load_from_checkpoint(os.path.join(
            'saved_models', 'retrieval', 'last.ckpt'
        ))
        self.sbir_model.apply(freeze_model)

        self.detection_model = WSDDN()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def forward(self, img, bboxes, query):
        '''
            img: [B, 3, H, W]
            bboxes: List of [N, 4]
            query: [B, C, 3, H, W]
        '''
        B, C, D, H, W = query.shape
        with torch.no_grad():
            query_feat = torch.stack([self.sbir_model(item, dtype='sketch') for item in query])
        
        # List of [N, C]
        all_combined_scores = self.detection_model(img, bboxes, query_feat)
        return all_combined_scores

    def training_step(self, batch, batch_idx):
        img_tensor, sk_tensor, proposals, target = batch
        all_combined_scores = self.forward(img_tensor, proposals, sk_tensor)

        loss = WSDDN.calculate_loss(all_combined_scores, target)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        img_tensor, sk_tensor, proposals, target = batch
        all_combined_scores = self.forward(img_tensor, proposals, sk_tensor)

        loss = WSDDN.calculate_loss(all_combined_scores, target)
        self.log('val_loss', loss)
        return all_combined_scores
