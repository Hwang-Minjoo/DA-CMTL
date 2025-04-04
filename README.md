# DA-CMTL 🧠🕸️🩸🍬
A domain-agnostic multi-task learning (DA-CMTL) model for generalized glucose level and hypoglycemia event prediction

## 📁 Repository Structure
📁 results/ # Directory for result files (not uploaded) 
📁 src/ # Reusable utility functions and model-related codes
📄 0_data_EDA.ipynb # Exploratory data analysis 
📄 1_data_preproc.py # Data preprocessing 
📄 2_train_base_model.py # Base model training 
📄 3_train_cl_model.py # DA-CMTL (continual learning) training 
📄 4_eval_personalized_model.py # Evaluation on personalized setting 
📄 5_tuning_real_model.py # Final model tuning 
📄 6_analysis_controlled_result.ipynb # Controlled result analysis 
📄 7_visualize_prediction_result.ipynb # Visualization of predictions

## ⚙️ Setup Instructions (Need to prepare datasets for running below source codes)
### 1. Clone and Create Environment
```bash
git clone https://github.com/[your_username]/DA-CMTL.git
cd DA-CMTL

# Create conda environment
conda create -n dacmtl python=3.10
conda activate dacmtl
```

### 2. Data Preprocessing
```bash
python 1_data_preproc.py --dataset_name {DATASET_NAME}
```

### 3. Train Models
```bash
# Train baseline model (PH: Prediciton Horizon)
python 2_train_base_model.py --dataset_name {DATASET_NAME} --PH {PH}

# Train DA-CMTL model
python 3_train_cl_model.py --cl_type {Naive/EWC} --order {12/21} --PH {PH}
```

### 4. Evaluation
```bash
python 4_eval_personalized_model.py --tgt_data {TARGET_DATA} --tuning_day {DAY_LENGTH}
python 5_tuning_real_model.py --tgt_data {TARGET_DATA} --tuning_day {DAY_LENGTH}
```
