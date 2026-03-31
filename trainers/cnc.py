import torch
import torch.nn as nn
from torch.nn import functional as F
import tqdm
import os

from models.model import CNC_FAS
from tools.evaluate import evaluate

class CNCTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        # Handle nested config structure: Use 'train' section for device/lr, or fallback to root
        self.train_cfg = cfg.get('train', cfg) 
        self.device = self.train_cfg.get('device', 'cuda')
        
        self.build_model()
        
    def build_model(self):
        print(f"Building CNC-FAS Model...")

        model_cfg = self.cfg.get('model', {}).copy()
        model_cfg['device'] = self.device

        self.model = CNC_FAS(model_cfg)

        prec = self.train_cfg.get('precision', 'fp32')
        if prec == 'fp32' or prec == 'amp':
            self.model.float()
        elif prec == 'fp16':
            self.model.half()
            
        self.model.to(self.device)
        
        # 4. Optimizer Setup
        # We optimize: Prompts, Fusion, INN, Decoder, AND Layer Projectors
        params = list(self.model.prompt_learner.parameters()) + \
                 list(self.model.fusion.parameters()) + \
                 list(self.model.inn.parameters()) + \
                 list(self.model.decoder.parameters()) + \
                 list(self.model.layer_projectors.parameters())
        
        lr = self.train_cfg.get('lr', 0.002)
        print(f"Optimizer LR: {lr}")

        opt_cfg = self.cfg.get('optimizer', {})
        opt_type = opt_cfg.get('type', 'sgd').lower()
        weight_decay = float(opt_cfg.get('weight_decay', 5e-4))
        print(f"Initializing Optimizer: {opt_type.upper()} | LR: {lr} | WD: {weight_decay}")

        # 3. Initialize based on type
        if opt_type == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif opt_type == 'adamw':
            self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif opt_type == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {opt_type}")
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.train_cfg.get('epochs', 20)
        )
        
        self.scaler = torch.amp.GradScaler('cuda') if prec == 'amp' else None

        # Load URD MLE Weighting
        self.lambda_mle = self.cfg.get('lambda_mle', 0.001)

    def compute_loss(self, output, label):
        """
        CNC Loss Logic.
        """
        layer_logits = output["layer_logits"] # List of {enc, dec}
        maps = output["anomaly_maps"]         # [B, L, H, W]
        # router_logits = output["router_logits"]
        log_prior = output["log_prior"]
        log_post = output["log_post"]
        
        # ======================================================================
        # 1. Distillation Loss (Reconstruction)
        # ======================================================================
        loss_distill = maps.mean()
        
        # ======================================================================
        # 2. Normality Constraint (Classification)
        # ======================================================================
        # Pushes features towards the "Live" text embedding and away from "Spoof"
        loss_cls_enc = torch.tensor(0.0, device=maps.device)
        loss_cls_dec = torch.tensor(0.0, device=maps.device)
        
        for item in layer_logits:
            loss_cls_enc += F.cross_entropy(item["enc"], label)
            loss_cls_dec += F.cross_entropy(item["dec"], label)
        
        loss_constraint = loss_cls_enc # + 0.1 * loss_cls_dec
        
        # ======================================================================
        # 3. URD Density Estimation Loss (MLE)
        # ======================================================================
        # router_probs = F.softmax(router_logits, dim=1)
        # importance = router_probs.sum(dim=0)
        # std_imp = torch.std(importance)
        # mean_imp = importance.mean() + 1e-6
        # loss_moe = (std_imp / mean_imp) ** 2
        loss_mle = -torch.mean(log_prior) - torch.mean(log_post)
        loss_mle /= self.model.vit_width
        
        # ======================================================================
        # Total Loss
        # ======================================================================
        total_loss = loss_distill + loss_constraint + self.lambda_mle * loss_mle
        
        return total_loss, {
            "loss": total_loss.item(),
            "distill": loss_distill.item(),
            "cls_enc": loss_cls_enc.item(),
            "cls_dec": loss_cls_dec.item(),
            # "moe": loss_moe.item()
            "mle": loss_mle.item()
        }

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        
        meters = {
            "loss": 0.0,
            "distill": 0.0,
            "cls_enc": 0.0,
            "cls_dec": 0.0,
            # "moe": 0.0
            "mle": 0.0
        }
        # pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        
        for batch in train_loader:
            imgs, labels = batch
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            
            with torch.amp.autocast("cuda", enabled=self.scaler is not None):
                output = self.model(imgs)
                loss, loss_dict = self.compute_loss(output, labels)
                
            self.optimizer.zero_grad()
                
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
                
            for k, v in loss_dict.items():
                meters[k] += v
            
        self.scheduler.step()
        
        # Calculate averages
        num_batches = len(train_loader)
        for k in meters:
            meters[k] /= num_batches
            
        return meters # Return dictionary instead of scalar
    
    def evaluate(self, val_loader):
        return evaluate(val_loader, self.model, self.device)

    def save_model(self, path):
        # Ensure directory exists before saving
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
            
        torch.save(self.model.state_dict(), path)