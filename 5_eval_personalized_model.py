import argparse, os, warnings, torch
import numpy as np
import torch
from src.model import MC_MTL_SGRU

from src.config import Config as c
from src.windowDataset import *
from tqdm import tqdm

from src.smoother import *
from src.metrics import *


warnings.filterwarnings(action='ignore')
def print_result(sid, Y, Y_prob, Y_pred, clf_thresh, cl_type):
    rmse = calculate_rmse(np.array(Y), np.array(Y_pred))
    sens, spec = get_sens_spec((np.array(Y)<70)*1, (np.array(Y_prob)>clf_thresh)*1)        
    tqdm.write("|   {:^8}  |   {:^12}  |  {:^8.4f}  |  {:^8.4f}  |  {:^8.4f}  |".format(sid, cl_type, rmse, sens, spec))
    tqdm.write("-"*72)

def main():
    parser = argparse.ArgumentParser(description="Process some data.")
    parser.add_argument('--cl_type', type=str, required=True, help="type of CL-Strategy: Naive/EWC")
    parser.add_argument('--order', type=str, required=True, help="Order of training multiple datsets: 12/21")
    parser.add_argument('--PH', type=int, required=True, help="Prediction Horizon: 30/60")
    parser.add_argument('--tgt_data', type=str, required=True, help="Name of Target Dataset")
    parser.add_argument('--tuning_day', type=int, required=True, help="Number of days for fine-tuning")
    args = parser.parse_args()

    cl_strategy = args.cl_type.lower()
    iw, ow = (args.PH//5)*2, (args.PH//5)
    if cl_strategy.startswith('sd'):
        first_order, second_order = args.order[0], args.order[0]
    else:
        first_order, second_order = args.order[0], args.order[1]
    print(f"[Options] cl: {cl_strategy}, order: Sim{first_order}->Sim{second_order}->{args.tgt_data}, tuning-duration: {args.tuning_day} days")
    
    # Load Dataset
    if args.tgt_data.lower() == 'ohiot1dm':
        test_fp = 'Datasets/OhioT1DM/Combined/test/cgm+ins+iob.pt'
    elif args.tgt_data.lower() == 'shanghait1dm':
        test_fp = 'Datasets/ShanghaiT1DM/Combined/cgm+ins+iob.pt'
    elif args.tgt_data.lower() == 'diatrend':
        test_fp = 'Datasets/DiaTrend/Combined/cgm+ins+iob.pt'

    print(f"[Load Target Dataset] {test_fp}")
    test_dataset = torch.load(test_fp)

    # Set Scaler based on second dataset
    print(f"* Get scaler from Sim_Data_{first_order}")
    src_X = torch.load(f'Datasets/Sim_Data_{first_order}/Processed/KS_Train_Data_CGM+IOB.pt')
    scaler, scaler_1d, scaled_hypo = get_scaler(src_X)
    
    # Test to each Real Patient
    print("-"*72); tqdm.write("|   {:^8}  |   {:^12}  |  {:^8}  |  {:^8}  |  {:^8}  |".format('SID', 'FT-Type', 'RMSE', 'Sens', 'Spec')); print("-"*72); 
    
    true, pred, prob, errors = [], [], [], []
    all_results = []
    for i in range(len(test_dataset)):
        # get 1 patient
        _data = test_dataset[i]; data_len = len(_data);
        sid = str(_data.SID.unique()[0])      
    
        # Scaling & Set Dataloader
        _data = _data.loc[:, ['date', 'CGM', 'IOB']].set_index('date').to_numpy()
        test_data = _data.copy() if (test_fp.split('/')[1]=='OhioT1DM') else _data[(args.tuning_day * 288):]
        smoothed_test_data = test_data.copy()
        # smoothed_test_data[:,0] = get_ks_data(test_data[:,0], c.pre_ks_value).squeeze()
        smoothed_test_data[:,0] = get_ks_data(test_data[:,0], c.ft_pre_ks_value).squeeze()

        # -- Save true cgm data
        true_cgm_data = np.array(test_data[(args.PH//5)+iw-1:, 0])
        
        # Normalization
        scaled_test_data = torch.tensor(scaler.transform(smoothed_test_data), dtype=torch.float32)
        test_loader = get_personal_loader(scaled_test_data, iw, ow, c.stride, c.ft_batch_size, train=False)

        # -- Frozen
        frozen_model_fp = f'Model/CL/{c.model_name}_{cl_strategy.upper()}{args.order}_I{iw}_O{ow}.pth'
        frozen_model = MC_MTL_SGRU(c.input_size, c.hidden_size, c.output_size, c.n_layers, c.dropout_rate).to(c.device)
        frozen_model.load_state_dict(torch.load(frozen_model_fp))
        
        frozen_model.train()
        Y, Y_pred, Y_prob, Y_error = [], [], [], []
        with torch.no_grad():
            for batch_X_test, batch_y_test in tqdm(test_loader, ncols=100, leave=False):
                batch_X_test = batch_X_test.to(c.device) 
                preds_reg, preds_clf = [],[]
                for i in range(c.mc_sample_nbr):
                    out_reg, out_clf = frozen_model(batch_X_test)
                    preds_reg.append(scaler_1d.inverse_transform(out_reg.cpu().detach().numpy().reshape(-1, 1)))
                    preds_clf.append(out_clf)

                pred_reg, error_reg = get_ks_data(np.array(preds_reg).mean(axis=0), c.post_ks_value), get_ks_data(np.array(preds_reg).std(axis=0), c.post_ks_value)
                preds_clf = torch.mean(torch.stack(preds_clf), dim=0)  # Average classification predictions over samples
                prob_clf = torch.sigmoid(preds_clf).detach().cpu().numpy().squeeze().reshape(-1, 1)
    
                Y_pred.extend(pred_reg)
                Y_prob.extend(prob_clf)
                Y_error.extend(error_reg)

        
        print_result(sid, true_cgm_data, Y_prob, Y_pred, 0.5, 'Frozen')
        N = len(Y_pred)
        results = pd.DataFrame({
                'SID': [sid] * N,  # SID column with the same value repeated for all entries
                'FT_Type': ['Frozen'] * N,
                'CL_Type': [f"{cl_strategy.upper()}_{args.order}"] * N,
                'true': true_cgm_data,
                'pred': np.array(Y_pred).reshape(-1),
                'prob': np.array(Y_prob).reshape(-1),
                'error': np.array(Y_error).reshape(-1),                
            })
        all_results.append(results)
        
        # -- Head-Tuning
        ht_model_dir = f"Model/CL_headtuned/{test_fp.split('/')[1]}/FT_{args.tuning_day}DAY/"
        ht_model_fn = f"{c.model_name}_{args.cl_type.upper()}{args.order}_I{iw}_O{ow}_{sid}.pth"
        ht_model = MC_MTL_SGRU(c.input_size, c.hidden_size, c.output_size, c.n_layers, c.dropout_rate).to(c.device)
        ht_model.load_state_dict(torch.load(ht_model_dir+ht_model_fn))

        ht_model.train()
        Y, Y_pred, Y_prob, Y_error = [], [], [], []
        with torch.no_grad():
            for batch_X_test, batch_y_test in tqdm(test_loader, ncols=100, leave=False):
                batch_X_test = batch_X_test.to(c.device) 
                preds_reg, preds_clf = [],[]
                for i in range(c.mc_sample_nbr):
                    out_reg, out_clf = ht_model(batch_X_test)
                    preds_reg.append(scaler_1d.inverse_transform(out_reg.cpu().detach().numpy().reshape(-1, 1)))
                    preds_clf.append(out_clf)

                pred_reg, error_reg = get_ks_data(np.array(preds_reg).mean(axis=0), c.post_ks_value), get_ks_data(np.array(preds_reg).std(axis=0), c.post_ks_value)
                preds_clf = torch.mean(torch.stack(preds_clf), dim=0)  # Average classification predictions over samples
                prob_clf = torch.sigmoid(preds_clf).detach().cpu().numpy().squeeze().reshape(-1, 1)
    
                Y_pred.extend(pred_reg)
                Y_prob.extend(prob_clf)
                Y_error.extend(error_reg)

        print_result(sid, true_cgm_data, Y_prob, Y_pred, 0.5, 'Head-Tuned')
        results = pd.DataFrame({
                'SID': [sid] * N,  # SID column with the same value repeated for all entries
                'FT_Type': ['Head-Tuned'] * N,
                'CL_Type': [f"{cl_strategy.upper()}_{args.order}"] * N,
                'true': true_cgm_data,
                'pred': np.array(Y_pred).reshape(-1),
                'prob': np.array(Y_prob).reshape(-1),
                'error': np.array(Y_error).reshape(-1),                
            })
        all_results.append(results)

        # -- Fully-Tuning
        ft_model_dir = f"Model/CL_fullytuned/{test_fp.split('/')[1]}/FT_{args.tuning_day}DAY/"
        ft_model_fn = f"{c.model_name}_{args.cl_type.upper()}{args.order}_I{iw}_O{ow}_{sid}.pth"
        ft_model = MC_MTL_SGRU(c.input_size, c.hidden_size, c.output_size, c.n_layers, c.dropout_rate).to(c.device)
        ft_model.load_state_dict(torch.load(ft_model_dir+ft_model_fn))

        ft_model.train()
        Y, Y_pred, Y_prob, Y_error = [], [], [], []
        with torch.no_grad():
            for batch_X_test, batch_y_test in tqdm(test_loader, ncols=100, leave=False):
                batch_X_test = batch_X_test.to(c.device) 
                preds_reg, preds_clf = [],[]
                for i in range(c.mc_sample_nbr):
                    out_reg, out_clf = ft_model(batch_X_test)
                    preds_reg.append(scaler_1d.inverse_transform(out_reg.cpu().detach().numpy().reshape(-1, 1)))
                    preds_clf.append(out_clf)

                pred_reg, error_reg = get_ks_data(np.array(preds_reg).mean(axis=0), c.post_ks_value), get_ks_data(np.array(preds_reg).std(axis=0), c.post_ks_value)
                preds_clf = torch.mean(torch.stack(preds_clf), dim=0)  # Average classification predictions over samples
                prob_clf = torch.sigmoid(preds_clf).detach().cpu().numpy().squeeze().reshape(-1, 1)
    
                Y.extend(scaler_1d.inverse_transform(batch_y_test.reshape(-1,1)).reshape(-1,1))
                Y_pred.extend(pred_reg)
                Y_prob.extend(prob_clf)
                Y_error.extend(error_reg)

        print_result(sid, true_cgm_data, Y_prob, Y_pred, 0.5, 'Fully-Tuned')
        results = pd.DataFrame({
                'SID': [sid] * N,  # SID column with the same value repeated for all entries
                'FT_Type': ['Fully-Tuned'] * N,
                'CL_Type': [f"{cl_strategy.upper()}_{args.order}"] * N,
                'true': true_cgm_data,
                'pred': np.array(Y_pred).reshape(-1),
                'prob': np.array(Y_prob).reshape(-1),
                'error': np.array(Y_error).reshape(-1),                
            })
        all_results.append(results)

    final_results_df = pd.concat(all_results, ignore_index=True)
    result_save_dir = f"Results/Sim2Real/{test_fp.split('/')[1]}/PH{args.PH}/FT_{args.tuning_day}DAY/"
    os.makedirs(result_save_dir, exist_ok=True)
    result_save_fp = f'result_{args.tgt_data}_{cl_strategy.upper()}{args.order}.csv'
    
    print(f"-> File path of results: {result_save_dir+result_save_fp}")
    final_results_df.to_csv(result_save_dir+result_save_fp, index=False)

if __name__ == "__main__":
    formatted_title = f"\033[1mTest Model to Real World\033[0m"
    print('-'*(len(formatted_title))); print(f"|   {formatted_title}   |");print('-'*(len(formatted_title)));
    main()
