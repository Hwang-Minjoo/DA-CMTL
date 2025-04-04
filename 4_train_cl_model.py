import argparse, os, warnings, torch
import numpy as np
import torch

from src.model import MC_MTL_SGRU
from src.ewc_regularizer import EWC
from src.config import Config as c
from src.windowDataset import *
from src.EarlyStopping import *

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter



warnings.filterwarnings(action='ignore')

def main():
    parser = argparse.ArgumentParser(description="Process some data.")
    parser.add_argument('--cl_type', type=str, required=True, help="type of CL-Strategy: Naive/EWC")
    parser.add_argument('--order', type=str, required=True, help="Order of training multiple datsets: 12/21")
    parser.add_argument('--PH', type=int, required=True, help="Prediction Horizon: 30/60")
    args = parser.parse_args()
    writer = SummaryWriter(f'runs/Train_CL_Model_{args.cl_type}{args.order}_PH{args.PH}')

    cl_strategy = args.cl_type.lower()
    iw, ow = (args.PH//5)*2, (args.PH//5)

    # -- Set Data Learning Order
    first_data, second_data = f'Sim_Data_{args.order[0]}', f'Sim_Data_{args.order[1]}'
    print(f"* cl: {cl_strategy}, order: {first_data}->{second_data}")

    # -- Load Base Model
    base_model_fp = f'Model/BaseModel/{c.model_name}_SD{args.order[0]}_I{iw}_O{ow}.pth'
    model = MC_MTL_SGRU(c.input_size, c.hidden_size, c.output_size, c.n_layers, c.dropout_rate).to(c.device)
    model.load_state_dict(torch.load(base_model_fp))
    print(f"* Base Model File Path: {base_model_fp}")

    # Set Scaler based on second dataset
    src_X = torch.load(f'Datasets/{first_data}/Processed/KS_Train_Data_CGM+IOB.pt')
    scaler, scaler_1d, scaled_hypo = get_scaler(src_X)
    src_train_loader = get_multiple_loader(src_X, iw, ow, c.stride, c.batch_size, scaler)

    # Load Data
    train_fp, valid_fp = f'Datasets/{second_data}/Processed/KS_Train_Data_CGM+IOB.pt', f'Datasets/{second_data}/Processed/KS_Valid_Data_CGM+IOB.pt'
    train_X, valid_X = torch.load(train_fp), torch.load(valid_fp)

    # Get Data Loader
    train_loader, valid_loader = get_multiple_loader(train_X, iw, ow, c.stride, c.batch_size, scaler), get_multiple_loader(valid_X, iw, ow, c.stride, c.batch_size, scaler)

    # Set Loss Function
    criterion_reg, criterion_clf = torch.nn.MSELoss(), torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor([c.clf_hypo_w]).to(c.device))

    # Setting for Learning Procedure
    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=c.lr*0.1)
    early_stopping = EarlyStopping(patience=c.patience)

    # Set EWC 
    ewc = EWC(model, src_train_loader, c.device, scaled_hypo)

    # Train Model with CL Strategy
    for epoch in tqdm(range(c.epochs), desc="Epochs", unit="epoch", ncols=100):
        train_loss, valid_loss, ewc_penalties = 0, 0, 0
        model.train()
        for batch_X, batch_y in tqdm(train_loader, desc="Training Batch", leave=False, ncols=100):
            optimizer.zero_grad()
            batch_X, batch_y = batch_X.to(c.device), batch_y[:,-1,0].to(c.device)
            batch_y_binary = (batch_y < scaled_hypo)*1.0
            train_out_reg, train_out_clf = model(batch_X)
            
            reg_loss = criterion_reg(train_out_reg.reshape(-1,1), batch_y.reshape(-1,1)) * c.reg_w
            clf_loss = criterion_clf(train_out_clf.reshape(-1,1), batch_y_binary.reshape(-1,1))

            ewc_penalty = ewc.penalty(model)
            if cl_strategy == 'ewc':
                loss = reg_loss + clf_loss + ewc_penalty
            else:
                loss = reg_loss + clf_loss
                
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss
            ewc_penalties += ewc_penalty
        train_loss/= len(train_loader); ewc_penalties/= len(train_loader);
        
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
    
        tqdm.write(f"Epoch #{epoch:02d}. Train: {train_loss:.5f}, Valid: {valid_loss:.5f}, EarlyStopping: {early_stopping.counter}, EWC: {ewc_penalties:.5f}")
        if early_stopping.early_stop:
            break

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Valid', valid_loss, epoch)
        writer.add_scalar('EWC Penalty', ewc_penalties, epoch)
    writer.close()

    print('-'*80)
    save_dir = f"Model/CL/"
    save_fn = f"{c.model_name}_{cl_strategy.upper()}{args.order[0]}{args.order[1]}_I{iw}_O{ow}.pth"

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_dir+save_fn)
    print(f'--> [Saved] {save_dir+save_fn}')
    

if __name__ == "__main__":
    formatted_title = f"\033[1mTrain Model with CL strategies\033[0m"
    print('-'*(len(formatted_title))); print(f"|   {formatted_title}   |");print('-'*(len(formatted_title)));
    main()
