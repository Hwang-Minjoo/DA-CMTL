import torch
class Config:
    # Device configuration
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # Training parameters
    epochs=200
    patience=10
    lr=0.001
    batch_size=128
    weight_decay=1.000000e-07
    model_name = 'MC_MTL_SGRU'

    ft_batch_size = batch_size // 2
    ft_lr = lr * 0.1
    
    # Model parameters
    input_size = 2
    output_size = 2 # 필요한가
    hidden_size = 128
    n_layers = 2
    dropout_rate = 0.3

    # Dataset
    stride = 1
    
    # Prediction parameters
    mc_sample_nbr = 20

    # Loss
    clf_hypo_w = 10
    reg_w = 100
    
    
    # Data parameters
    sim_d1_train_fp = 'Datasets/Sim_Data_1/Processed/KS_Train_Data_CGM+IOB.pt'
    sim_d1_valid_fp = 'Datasets/Sim_Data_1/Processed/KS_Valid_Data_CGM+IOB.pt'
    sim_d2_train_fp = 'Datasets/Sim_Data_2/Processed/KS_Train_Data_CGM+IOB.pt'
    sim_d2_valid_fp = 'Datasets/Sim_Data_2/Processed/KS_Valid_Data_CGM+IOB.pt'

    # Smoothing
    pre_ks_value = 0.1
    post_ks_value = 0.3
    ft_pre_ks_value = 0.01
    
    