import os
import yaml
import sys
import time
from utils.logger import Logger
# Import your config loader
from utils.config import load_config 
# Import your trainer
from trainers.cnc import CNCTrainer 
from dataloader import FASDataset
from torch.utils.data import DataLoader

def get_time_str(start_time):
    """Formats elapsed time as 'X min Y sec'"""
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    return f"{minutes} min {seconds} sec"

def print_header():
    """Prints the table header"""
    print("-" * 110)
    print(f"{' ' * 36}** Starting CNC-FAS Model Training! **")
    print("-" * 110)
    # Header Structure: 
    # Valid: APCER, BPCER, HTER, AUC
    # Train: LR, Distill, Const, MoE, Total
    h_epoch = "| epoch |"
    h_valid = f" { 'VALID':^26} |"
    h_train = f" { 'Train':^36} |"
    h_best  = f" {'Current Best':^14} |"
    h_time  = f" {'time':^13} |"
    print(h_epoch + h_valid + h_train + h_best + h_time)    
    
    sub_valid = "APCER  BPCER   HTER    AUC"   # 5 + 2 + 5 + 2 + 5 + 2 + 5 = 26
    sub_train = "   lr     L_dis  L_con  L_mle  L_tot" # 8 + 2 + 5 + 2 + 5 + 2 + 5 + 2 + 5 = 36
    sub_best  = " HTER     AUC "               # 5 + 4 + 5 = 14
    print(f"|       | {sub_valid} | {sub_train} | {sub_best} |               |")
    print("-" * 110)

def main():
    # 1. Load Config (Merging args + yaml + generating exp name)
    cfg = load_config()
    
    # 2. Setup Directories
    # Structure: output_dir / experiment_name /
    root_output_dir = cfg['train']['output_dir']
    exp_name = cfg['name']
    
    exp_save_path = os.path.join(root_output_dir, exp_name)
    os.makedirs(exp_save_path, exist_ok=True)
    
    # Update config with the specific experiment path for the trainer to use later
    cfg['exp_save_path'] = exp_save_path

    # 3. Save Current Config (exp_name.yaml)
    cfg_filename = f"{cfg['dataset']['source']}_to_{cfg['dataset']['target']}.yaml"
    cfg_save_path = os.path.join(exp_save_path, cfg_filename)

    with open(cfg_save_path, 'w', encoding='utf-8') as file:
        yaml.dump(cfg, file, allow_unicode=True, default_flow_style=False)
    
    print(f"Config saved to: {cfg_save_path}")

    # 4. Save Logs (exp_name_log.txt)
    log_filename = f"{cfg['dataset']['source']}_to_{cfg['dataset']['target']}_log.txt"
    log_save_path = os.path.join(exp_save_path, log_filename)
    
    sys.stdout = Logger()
    sys.stdout.open(log_save_path)
    
    # print(f"Experiment started: {exp_name}")
    print(f"Logging to: {log_save_path}")
    print("-" * 30)

    # 5. Initialize Data & Trainer
    print(f"Loading Source: {cfg['dataset']['source']}")
    train_set = FASDataset(
        root_dir=cfg['dataset']['root_path'], 
        protocol=cfg['dataset']['source'],
        live_only=True,
    )
    
    train_loader = DataLoader(
        train_set, 
        batch_size=cfg['train']['batch_size'], 
        shuffle=True, 
        num_workers=cfg['dataset']['num_workers'],
        pin_memory=True
    )

    print(f"Loading Target: {cfg['dataset']['target']}")
    val_set = FASDataset(
        root_dir=cfg['dataset']['test_path'], 
        protocol=cfg['dataset']['target'],
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=cfg['train']['batch_size'], 
        shuffle=False, 
        num_workers=cfg['dataset']['num_workers'],
        pin_memory=True
    )

    trainer = CNCTrainer(cfg)

    best_hter = float('inf')
    best_auc = 0.0
    start_time = time.time()

    print_header()
    
    # 6. Training Loop
    for epoch in range(1, cfg['train']['epochs'] + 1):
        loss_dict = trainer.train_epoch(train_loader, epoch)
        # print(f"Epoch {epoch} | Loss: {loss:.5f}")

        if epoch % cfg['train']['eval_every'] == 0 or epoch == cfg['train']['epochs']:
            metrics = trainer.evaluate(val_loader)

            is_best = metrics['HTER'] < best_hter or (metrics['HTER'] == best_hter and metrics['AUC'] > best_auc)
            if is_best:
                best_hter = metrics['HTER']
                best_auc = metrics['AUC']
            
            current_lr = trainer.optimizer.param_groups[0]['lr']
            valid_str = f"{metrics['APCER']*100:5.2f}  {metrics['BPCER']*100:5.2f}  {metrics['HTER']*100:5.2f}  {metrics['AUC']*100:5.2f}"
        
            # Train: lr, L_dist, L_cons, L_mle, L_total
            # Note: L_cons = cls_enc + 0.1*cls_dec. Let's just print cls_enc or sum them for brevity.
            # Let's use L_cons derived roughly from the dict or just print components.
            l_cons = loss_dict['cls_enc'] + 0.1 * loss_dict['cls_dec']
            train_str = f"{current_lr:.6f}  {loss_dict['distill']:.3f}  {l_cons:.3f}  {loss_dict['mle']:.3f}  {loss_dict['loss']:.3f}"
            
            # Best: HTER, AUC
            best_str = f"{best_hter*100:5.2f}    {best_auc*100:5.2f}"
            
            # Time
            time_str = get_time_str(start_time)
            
            # 6. Print Row
            print(f"|  {epoch:2d}   | {valid_str} | {train_str} | {best_str} | {time_str:^13} |")
        
        # Save checkpoints into the experiment folder
        if cfg['train']['save_ckpt'] and epoch % 5 == 0:
            ckpt_path = os.path.join(exp_save_path, f"epoch_{epoch}.pth")
            trainer.save_model(ckpt_path)

if __name__ == "__main__":
    main()