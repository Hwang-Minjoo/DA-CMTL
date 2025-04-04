import argparse, os, warnings, torch
import numpy as np
import torch

from src.model import MC_MTL_SGRU
from src.ewc_regularizer import EWC
from src.config import Config as c
from src.windowDataset import *
from src.EarlyStopping import *
from src.smoother import *

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

warnings.filterwarnings(action='ignore')

def main():
    parser = argparse.ArgumentParser(description="Process some data.")
    parser.add_argument('--cl_type', type=str, required=True, help="type of CL-Strategy: Naive/EWC")
    parser.add_argument('--order', type=str, required=True, help="Order of training multiple datsets: 12/21")
    parser.add_argument('--PH', type=int, required=True, help="Prediction Horizon: 30/60")
    parser.add_argument('--tgt_data', type=str, required=True, help="Name of Target Dataset")
    parser.add_argument('--tuning_day', type=int, required=True, help="Number of days for fine-tuning")
    args = parser.parse_args()
    
    writer = SummaryWriter(f'runs/Tuning_CL_Model_{args.cl_type}{args.order}_PH{args.PH}')

    # -- Define Model File Path
    iw, ow = (args.PH//5)*2, (args.PH//5)
    cl_strategy = args.cl_type.lower()
    if cl_strategy.startswith('sd'):
        model_fp = f'Model/BaseModel/{c.model_name}_SD{args.order}_I{iw}_O{ow}.pth'
        first_order, second_order = args.order[0], args.order[0]
    else:
        model_fp = f'Model/CL/{c.model_name}_{cl_strategy.upper()}{args.order}_I{iw}_O{ow}.pth'
        first_order, second_order = args.order[0], args.order[1]
    print(f"[Options] cl: {cl_strategy}, order: Sim{first_order}->Sim{second_order}, tuning-duration: {args.tuning_day} days")
    print(f"* Get pretrained model's file path: {model_fp}")

    # Get Target Dataset
    if args.tgt_data.lower() == 'ohiot1dm':
        test_fp = 'Datasets/OhioT1DM/Combined/train/cgm+ins+iob.pt'
    elif args.tgt_data.lower() == 'shanghait1dm':
        test_fp = 'Datasets/ShanghaiT1DM/Combined/cgm+ins+iob.pt'
    elif args.tgt_data.lower() == 'diatrend':
        test_fp = 'Datasets/DiaTrend/Combined/cgm+ins+iob.pt'
    
    test_dataset = torch.load(test_fp)
    print(f"* Get target dataset: {args.tgt_data}, {test_fp}")

    # -- Set Scaler
    print(f"* Get scaler from Sim_Data_{first_order}")
    src_X = torch.load(f'Datasets/Sim_Data_{first_order}/Processed/KS_Train_Data_CGM+IOB.pt')
    scaler, scaler_1d, scaled_hypo = get_scaler(src_X)

    # Setting for EWC
    print(f"* Get EWC Information from Sim_Data_{second_order}")
    second_X = torch.load(f'Datasets/Sim_Data_{second_order}/Processed/KS_Train_Data_CGM+IOB.pt')
    src_train_loader = get_multiple_loader(second_X, iw, ow, c.stride, c.batch_size, scaler)
    
    # Setting for Learning Procedure
    criterion_reg, criterion_clf = torch.nn.MSELoss(), torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor([c.clf_hypo_w]).to(c.device))

    print(f"* Fine-Tuning")
    for i in range(len(test_dataset)):
        _data = test_dataset[i]
        sid = _data.SID.unique()[0]
        print(f"[{i+1}] Tuning {sid}")
    
        # -- Training model with head-tuning method
        tuning_duration = args.tuning_day * 288
        train_len = int(tuning_duration * 0.7)
        valid_len = tuning_duration-train_len

        if args.tgt_data.lower() == 'ohiot1dm':
            train_data = _data.iloc[-tuning_duration:-valid_len][['date','CGM','IOB']].set_index('date').to_numpy()
            valid_data = _data.iloc[-valid_len:][['date','CGM','IOB']].set_index('date').to_numpy()
        else:
            train_data = _data.iloc[:train_len][['date','CGM','IOB']].set_index('date').to_numpy()
            valid_data = _data.iloc[train_len:tuning_duration][['date','CGM','IOB']].set_index('date').to_numpy()
    
         # Apply Smoothing for tuning
        print(f"\t- Smoothing with coeff:{c.pre_ks_value}")
        smoothed_train_data, smoothed_valid_data = train_data.copy(), valid_data.copy()
        smoothed_train_data[:,0], smoothed_valid_data[:,0] = get_ks_data(train_data[:,0], c.pre_ks_value).squeeze(), get_ks_data(valid_data[:,0], c.pre_ks_value).squeeze()

        # Apply Scaling & Get Loader
        scaled_train_data = torch.tensor(scaler.transform(smoothed_train_data), dtype=torch.float32)
        scaled_valid_data = torch.tensor(scaler.transform(smoothed_valid_data), dtype=torch.float32)
        print(f"\t- Train: {train_len/288:.1f} days, Valid: {valid_len/288:.1f} days")
        
        train_loader,valid_loader = get_personal_loader(scaled_train_data, iw, ow, c.stride, c.ft_batch_size, train=True), get_personal_loader(scaled_valid_data, iw, ow, c.stride, c.ft_batch_size, train=False)
        
        #########################################
        #             Head-Tuning               #
        #########################################
        print(f"\t- Head-Tuning")
        ht_model = MC_MTL_SGRU(c.input_size, c.hidden_size, c.output_size, c.n_layers, c.dropout_rate).to(c.device)
        ht_model.load_state_dict(torch.load(model_fp))

        # Setting for Continual Learning
        ewc = EWC(ht_model, src_train_loader, c.device, scaled_hypo)
        
        for name, param in ht_model.named_parameters():
            if 'fc' not in name: 
                param.requires_grad = False
            elif 'fc' in name: 
                param.requires_grad = True

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, ht_model.parameters()), lr=c.ft_lr, weight_decay=c.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=c.ft_lr*0.1)
        early_stopping = EarlyStopping(patience=c.patience) 
        
        train_loss_arr, valid_loss_arr = [],[]       
        ht_model.train()
        for epoch in tqdm(range(c.epochs), desc="Epochs", unit="epoch", ncols=100):
            train_loss, valid_loss = 0, 0
            for batch_X, batch_y in tqdm(train_loader, desc="Training Batch", leave=False, ncols=100):
                batch_X, batch_y = batch_X.to(c.device), batch_y.to(c.device)
                batch_y_binary = (batch_y < scaled_hypo)*1.0
                
                train_out_reg, train_out_clf = ht_model(batch_X)
    
                reg_loss = criterion_reg(train_out_reg.reshape(-1,1), batch_y.reshape(-1,1)) * c.reg_w
                clf_loss = criterion_clf(train_out_clf.reshape(-1,1), batch_y_binary.reshape(-1,1))
                ewc_penalty = ewc.penalty(ht_model)
                if cl_strategy == 'ewc':
                    loss = reg_loss + clf_loss + ewc_penalty
                else:
                    loss = reg_loss + clf_loss
                train_loss += loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            train_loss /= len(train_loader)
    
            with torch.no_grad():
                for batch_X, batch_y in tqdm(valid_loader, desc="Validation Batch", leave=False, ncols=100):
                    batch_X, batch_y = batch_X.to(c.device), batch_y.to(c.device)
                    batch_y_binary = (batch_y < scaled_hypo)*1.0
    
                    valid_out_reg, valid_out_clf = ht_model(batch_X)
                    
                    valid_reg_loss = criterion_reg(valid_out_reg.reshape(-1,1), batch_y.reshape(-1,1)) * c.reg_w
                    valid_clf_loss = criterion_clf(valid_out_clf.reshape(-1,1), batch_y_binary.reshape(-1,1))
                
                    valid_loss += (valid_reg_loss + valid_clf_loss)                
                valid_loss /= len(valid_loader)
            early_stopping(valid_loss, ht_model)
            if early_stopping.early_stop:
                break

        save_dir = f"Model/CL_headtuned/{test_fp.split('/')[1]}/FT_{args.tuning_day}DAY/"
        save_fn = f"{c.model_name}_{cl_strategy.upper()}{args.order[0]}{args.order[1]}_I{iw}_O{ow}_{sid}.pth"
    
        os.makedirs(save_dir, exist_ok=True)
        torch.save(ht_model.state_dict(), save_dir+save_fn)
        print(f'\t--> [Saved] {save_dir+save_fn}')

        #########################################
        #             Fully-Tuning              #
        #########################################
        print(f"\t- Fully-Tuning")
        ft_model = MC_MTL_SGRU(c.input_size, c.hidden_size, c.output_size, c.n_layers, c.dropout_rate).to(c.device)
        ft_model.load_state_dict(torch.load(model_fp))

        # Setting for Continual Learning
        ewc = EWC(ft_model, src_train_loader, c.device, scaled_hypo)
        
        optimizer = torch.optim.Adam(ft_model.parameters(), lr = c.ft_lr, weight_decay=c.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=c.ft_lr*0.1)
        early_stopping = EarlyStopping(patience=c.patience) 
        
        train_loss_arr, valid_loss_arr = [],[]       
        ft_model.train()
        for epoch in tqdm(range(c.epochs), desc="Epochs", unit="epoch", ncols=100):
            train_loss, valid_loss = 0, 0
            for batch_X, batch_y in tqdm(train_loader, desc="Training Batch", leave=False, ncols=100):
                batch_X, batch_y = batch_X.to(c.device), batch_y.to(c.device)
                batch_y_binary = (batch_y < scaled_hypo)*1.0
                
                train_out_reg, train_out_clf = ft_model(batch_X)
    
                reg_loss = criterion_reg(train_out_reg.reshape(-1,1), batch_y.reshape(-1,1)) * c.reg_w
                clf_loss = criterion_clf(train_out_clf.reshape(-1,1), batch_y_binary.reshape(-1,1))
                ewc_penalty = ewc.penalty(ft_model)
                if cl_strategy == 'ewc':
                    loss = reg_loss + clf_loss + ewc_penalty
                else:
                    loss = reg_loss + clf_loss
                train_loss += loss
    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            train_loss /= len(train_loader)
    
            with torch.no_grad():
                for batch_X, batch_y in tqdm(valid_loader, desc="Validation Batch", leave=False, ncols=100):
                    batch_X, batch_y = batch_X.to(c.device), batch_y.to(c.device)
                    batch_y_binary = (batch_y < scaled_hypo)*1.0
    
                    valid_out_reg, valid_out_clf = ft_model(batch_X)
                    valid_reg_loss = criterion_reg(valid_out_reg.reshape(-1,1), batch_y.reshape(-1,1)) * c.reg_w
                    valid_clf_loss = criterion_clf(valid_out_clf.reshape(-1,1), batch_y_binary.reshape(-1,1))

                    loss = valid_reg_loss + valid_clf_loss
                    valid_loss += loss
            valid_loss /= len(valid_loader)
            early_stopping(valid_loss, ft_model)
            if early_stopping.early_stop:
                break

        save_dir = f"Model/CL_fullytuned/{test_fp.split('/')[1]}/FT_{args.tuning_day}DAY/"
        save_fn = f"{c.model_name}_{cl_strategy.upper()}{args.order[0]}{args.order[1]}_I{iw}_O{ow}_{sid}.pth"
    
        os.makedirs(save_dir, exist_ok=True)
        torch.save(ft_model.state_dict(), save_dir+save_fn)
        print(f'\t--> [Saved] {save_dir+save_fn}')
        print('-'*100)
    
if __name__ == "__main__":
    formatted_title = f"\033[1mTrain Model with CL strategies\033[0m"
    print('-'*(len(formatted_title))); print(f"|   {formatted_title}   |");print('-'*(len(formatted_title)));
    main()
