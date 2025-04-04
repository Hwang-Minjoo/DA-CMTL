import argparse, os, warnings, torch
warnings.filterwarnings(action='ignore')

import numpy as np

import torch
import torch.nn as nn

from src.model import MC_MTL_SGRU
from src.windowDataset import *
from src.EarlyStopping import *
from src.config import Config as c

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, help="Name of Dataset")
    parser.add_argument('--PH', type=str, required=True, help="Prediction Horizon")
    args = parser.parse_args()

    ow = int(args.PH)//5; iw = ow *2
    print(f"* Dataset: {args.dataset_name}\n* Model: {c.model_name}\n* N_Layers: {c.n_layers}\n* PH: {int(args.PH)}\n")
    writer = SummaryWriter(f'runs/Train_Base_Model_{args.dataset_name}')
    
    print('-'*80)
    # Load Data
    train_fp, valid_fp = f'Datasets/{args.dataset_name}/Processed/KS_Train_Data_CGM+IOB.pt', f'Datasets/{args.dataset_name}/Processed/KS_Valid_Data_CGM+IOB.pt'
    train_X, valid_X = torch.load(train_fp), torch.load(valid_fp)

    # Set Scaler
    scaler, scaler_1d, scaled_hypo = get_scaler(train_X)
    train_loader, valid_loader = get_multiple_loader(train_X, iw, ow, c.stride, c.batch_size, scaler), get_multiple_loader(valid_X, iw, ow, c.stride, c.batch_size, scaler)

    # Set Loss Function
    criterion_reg, criterion_clf = torch.nn.MSELoss(), torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor([c.clf_hypo_w]).to(c.device))

    # Setting for Learning Procedure
    model = MC_MTL_SGRU(c.input_size, c.hidden_size, c.output_size, c.n_layers, c.dropout_rate).to(c.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=c.lr*0.1)
    early_stopping = EarlyStopping(patience=c.patience)

    
    # Model Train
    for epoch in tqdm(range(c.epochs), desc="Epochs", unit="epoch", ncols=100):
        train_loss, valid_loss = 0, 0
        model.train()
        for batch_X, batch_y in tqdm(train_loader, desc="Training Batch", leave=False, ncols=100):
            optimizer.zero_grad()
            batch_X, batch_y = batch_X.to(c.device), batch_y[:,-1,0].to(c.device)
            batch_y_binary = (batch_y < scaled_hypo)*1.0
            train_out_reg, train_out_clf = model(batch_X)
            
            reg_loss = criterion_reg(train_out_reg.reshape(-1,1), batch_y.reshape(-1,1)) * c.reg_w
            clf_loss = criterion_clf(train_out_clf.reshape(-1,1), batch_y_binary.reshape(-1,1))
            loss = reg_loss + clf_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss
        train_loss/= len(train_loader)
        
        with torch.no_grad():
            for batch_X, batch_y in tqdm(valid_loader, desc="Validation Batch", leave=False, ncols=100):
                batch_X, batch_y = batch_X.to(c.device), batch_y[:,-1,0].to(c.device)
                batch_y_binary = (batch_y < scaled_hypo)*1.0
                valid_out_reg, valid_out_clf = model(batch_X)
    
                valid_reg_loss = criterion_reg(valid_out_reg.reshape(-1,1), batch_y.reshape(-1,1)) * c.reg_w
                valid_clf_loss = criterion_clf(valid_out_clf.reshape(-1,1), batch_y_binary.reshape(-1,1))
    
                valid_loss += (valid_reg_loss + valid_clf_loss)     
        valid_loss /= len(valid_loader)
        early_stopping(valid_loss, model)
    
        tqdm.write(f"Epoch #{epoch:02d}. Train: {train_loss:.5f}, Valid: {valid_loss:.5f}, EarlyStopping: {early_stopping.counter}")
        if early_stopping.early_stop:
            break

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Valid', valid_loss, epoch)
    writer.close()

    print('-'*80)
    save_dir = f"Model/BaseModel/"
    save_fn = f"{c.model_name}_SD{args.dataset_name[-1]}_I{iw}_O{ow}.pth"

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_dir+save_fn)
    print(f'--> [Saved] {save_dir+save_fn}')


if __name__ == "__main__":
    formatted_title = f"\033[1mTrain Base Model\033[0m"
    print('-'*(len(formatted_title))); print(f"|   {formatted_title}   |");print('-'*(len(formatted_title))); 
    
    main()
    
    
