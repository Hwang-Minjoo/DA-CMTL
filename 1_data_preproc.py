import argparse, os, warnings, torch
import pandas as pd
import numpy as np
from openpyxl import load_workbook
import xml.etree.ElementTree as ET
from datetime import datetime

warnings.filterwarnings(action='ignore')

from src.iob_model import get_iob
from src.smoother import *

# Function to remove each day which has more than 20 min consecutive CGM missings
def remove_consecutive_nans(df, column, n, date_column):
    df = df.reset_index()
    df['group'] = (df[column].notna().cumsum())
    df['nan_count'] = df.groupby('group')[column].transform(lambda x: x.isna().sum())
    
    days_missing = df[(df.nan_count>=n+1) & (pd.isna(df[column]))][date_column].unique()
    df_cleaned = df[~df[date_column].isin(days_missing)]
    
    return df_cleaned

# Function to filter the DataFrame by valid dates
def filter_by_date(df, valid_dates, date_column):
    return df[df[date_column].dt.floor('d').isin(valid_dates)]

def main():
    parser = argparse.ArgumentParser(description="Process some data.")
    parser.add_argument('--dataset_name', type=str, required=True, help="Name of the dataset to process")
    args = parser.parse_args()
    print(f"[Data Preprocessing..] {args.dataset_name}")
    
    raw_dir = f'Datasets/{args.dataset_name}/Raw/'
    print('Original File Path: ', args.dataset_name, raw_dir)

    #####################################
    #         Simulator Dataset         #
    #####################################
    if args.dataset_name.lower().startswith('sim'):
        file_list = os.listdir(raw_dir)
        print(file_list)
        df = pd.DataFrame(columns=["Time","BG","CGM","CHO","insulin","LBGI","HBGI","Risk","IOB","S_ID"])        
        cnt=1
        for i, file in enumerate(file_list):
            if (os.path.splitext(file)[1]=='.csv') and (file.startswith('adult')):
                print(cnt, file); cnt+=1
                data = pd.read_csv(raw_dir + file)
                data['Time'] = pd.to_datetime(data.Time)
                sid = f"adult#{int(file.split('#')[1][1:3]):02d}"
                
                # -- CGM Missing: Cubic Interpolation
                interp_data = data.copy()
                interp_data['CGM'].replace(0, np.nan, inplace=True)
                interp_data['CGM'] = interp_data['CGM'].interpolate(method='cubic')
    
                # -- CGM Outlier: 40-400 mg/dL
                filtered_data = interp_data.copy()
                filtered_data.loc[interp_data.CGM<40, 'CGM'] = 40
                filtered_data.loc[interp_data.CGM>400, 'CGM'] = 400
                filtered_data['S_ID'] = sid
                
                df = pd.concat([df,filtered_data])
                
        new_dir = f"{raw_dir.replace('Raw', 'Processed')}"
        df = df.reset_index(drop=True)
        df.to_csv(f"{new_dir}Data_CGM+IOB+INS.csv")

            
        data_numpy = np.array(df.loc[:, ['CGM', 'IOB', 'insulin']])
        data_tensor = torch.tensor(np.array([df.loc[df.S_ID == sid, ['CGM', 'IOB', 'insulin']].rename(columns = {'insulin':'INS'}) for sid in sorted(df.S_ID.unique())]), dtype=torch.float32)
    
        new_fn = f"Data_CGM+IOB+INS.pt"
        torch.save(data_tensor, new_dir+new_fn)   
        
        # -- Data Split: Train vs Valid = 8:2
        np.random.seed(14)
        N = len(data_tensor); N_train = int(N*0.8); N_valid = N-N_train;
        idx_list = np.arange(N)
        selected_train_idx = np.random.choice(idx_list, size=N_train, replace=False)
        selected_valid_idx = list(set(idx_list) - set(selected_train_idx))

        new_train_fn, new_valid_fn = new_fn.replace('Data', 'Train_Data'),new_fn.replace('Data', 'Valid_Data')
        torch.save(data_tensor[selected_train_idx], f"{new_dir + new_train_fn}")
        torch.save(data_tensor[selected_valid_idx], f"{new_dir + new_valid_fn}")
        
        print('-'*60)
        print(f"[New File Path]\n{new_dir+new_fn}\n{new_dir}{new_fn.replace('Data', 'Train_Data')}\n{new_dir}{new_fn.replace('Data', 'Valid_Data')}")
        print('-'*60)

        print(f"[Offline Smoothing to {args.dataset_name}]", end = ' ')
        for fn in os.listdir(new_dir):
            if fn.endswith('.pt')&(fn.startswith('KS')==False):
                new_fp = new_dir + fn
                print(f"Target File Name: {new_fp}")
                data = torch.load(new_fp).numpy()
                ks_data = np.zeros_like(data)
                for i in range(ks_data.shape[0]):
                    ks_data[i,:,0] = get_ks_data(data[i,:,0]).squeeze()
                ks_data[:,:,1] =  data[:,:,1]
                
                save_fn = f"{new_dir}KS_{fn}"
                print(f"{len(ks_data)} Patients\n--> [Saved] New File Name: {save_fn}")
                torch.save(ks_data, save_fn)
    
    #####################################
    #             OhioT1DM              #
    #####################################
    elif args.dataset_name.lower() == 'ohiot1dm':
        dir_train =  "Datasets/OhioT1DM/Raw/train/" 
        dir_test =  "Datasets/OhioT1DM/Raw/test/" 
        
        train_file_list = os.listdir(dir_train)
        test_file_list = os.listdir(dir_test)
        
        dir_list = [dir_train, dir_test]
        
        # -- Convert .xml to .csv
        date_column = 'ts'
        cgm_column = 'cgm'
        basal_column = 'basal'
        
        for dir_i, file_list in enumerate([train_file_list, test_file_list]):
            dir_fp = dir_list[dir_i]
            print(f"\033[1m\nProcessing {dir_fp}\033[0m")
            
            file_cnt = 0
            for i, filename in enumerate(file_list):
                if filename.endswith('.xml') == False:
                    continue
                    
                file_path = dir_fp+filename
                print(f"{file_cnt+1} Original File Path: {file_path}"); file_cnt+=1;
                
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                data = {}
                
                patient_id = root.attrib.get('id')
                weight = root.attrib.get('weight')
                insulin_type = root.attrib.get('insulin_type')
        
                
                '''
                Process CGM
                '''
                cgm_var = root.find('glucose_level')
                cgm_data = []
                if cgm_var is not None:
                    events = cgm_var.findall('event')
                    for event in events:
                        cgm_data.append({'ts': pd.to_datetime(event.attrib.get('ts'), dayfirst=True), 'cgm': float(event.attrib.get('value'))})
                cgm_df = pd.DataFrame(cgm_data)
                cgm_df['ts'] = pd.to_datetime(cgm_df['ts'], dayfirst=True)
                
                cgm_df.sort_values(by=date_column, inplace=True)  
                cgm_df[date_column] = cgm_df[date_column].dt.round('5min')
                cgm_df.set_index(date_column, inplace=True)
                cgm_df = cgm_df.resample('5min').first()
                
                not_missed_cgm_df = remove_consecutive_nans(cgm_df.reset_index(), cgm_column, 4, date_column).set_index(date_column)
                
                not_missed_cgm_df['is_interpolated'] = False
                not_missed_cgm_df.loc[pd.isna(not_missed_cgm_df[cgm_column]), 'is_interpolated'] = True
                
                interp_cgm_df = not_missed_cgm_df.copy()
                interp_cgm_df[cgm_column] = not_missed_cgm_df[cgm_column].interpolate(method = 'cubic')
                cgm_df = interp_cgm_df.loc[:, ['cgm', 'is_interpolated']].reset_index()

                cgm_df.loc[cgm_df['cgm']<40, 'cgm'] = 40
                cgm_df.loc[cgm_df['cgm']>400, 'cgm'] = 400
        
                '''
                Process Basal
                '''
                
                basal_var = root.find('basal')
                basal_data = []
                if basal_var is not None:
                    events = basal_var.findall('event')
                    for event in events:
                        basal_data.append({'ts': pd.to_datetime(event.attrib.get('ts'), dayfirst=True), 'basal': float(event.attrib.get('value'))})
                basal_df = pd.DataFrame(basal_data)
        
                basal_df[date_column] = basal_df[date_column].dt.round('5min')
                basal_df[basal_column] = basal_df[basal_column].fillna(0)  # Fill NaN values in 'Rate' column with 0
                
                basal_df = basal_df.sort_values(by=date_column)
                basal_df.set_index(date_column, inplace=True)
                
                basal_df = basal_df.resample('5min').first()
                basal_df[basal_column] = basal_df[basal_column].ffill()
                basal_df[basal_column] = basal_df[basal_column] / 12
                basal_df.reset_index(inplace=True)
                
        
                temp_basal_var = root.find('temp_basal')
                temp_basal_data = []
                if temp_basal_var is not None:
                    events = temp_basal_var.findall('event')
                    for event in events:
                        temp_basal_data.append({'ts_begin': pd.to_datetime(event.attrib.get('ts_begin'), dayfirst=True), 'ts_end': pd.to_datetime(event.attrib.get('ts_end'), dayfirst=True), 'temp_basal': float(event.attrib.get('value'))})
                    temp_basal_df = pd.DataFrame(temp_basal_data)
        
                    if not temp_basal_df.empty:
                        temp_basal_df['ts_begin'] = temp_basal_df['ts_begin'].dt.round('5min')
                        temp_basal_df['temp_basal'] = temp_basal_df['temp_basal'].fillna(0)  # Fill NaN values in 'Rate' column with 0
            
                        for _, temp_basal in temp_basal_df.iterrows():
                            basal_value = 0 if temp_basal.temp_basal==0 else temp_basal.temp_basal/12                    
                            basal_df.loc[(temp_basal.ts_begin <= basal_df.ts) & (temp_basal.ts_end >= basal_df.ts), 'basal'] = basal_value
                
                
                combined_df = pd.merge(cgm_df, basal_df.loc[:, [date_column, 'basal']], on=date_column, how='left')
                
                '''
                Process Bolus
                '''
                bolus_var = root.find('bolus')
                bolus_data = []
                if bolus_var is not None:
                    events = bolus_var.findall('event')
                    for event in events:
                        bolus_data.append({'ts_begin': pd.to_datetime(event.attrib.get('ts_begin'), dayfirst=True), 'ts_end': pd.to_datetime(event.attrib.get('ts_end'), dayfirst=True), 'bolus': float(event.attrib.get('dose')), 'type': event.attrib.get('type')})
                bolus_df = pd.DataFrame(bolus_data)
                bolus_df['duration_5min'] = [diff.total_seconds()/(60*5) for diff in bolus_df.ts_end - bolus_df.ts_begin]
                bolus_df['ts_begin'], bolus_df['ts_end'] = bolus_df['ts_begin'].dt.round('5min'), bolus_df['ts_end'].dt.round('5min')
        
                combined_df['bolus'] = 0.0
                for i, bolus in bolus_df.iterrows():
                    bolus_value = bolus.bolus / (bolus.duration_5min) if bolus.duration_5min != 0 else bolus.bolus
                    combined_df.loc[(bolus.ts_begin <= combined_df.ts) & (bolus.ts_end >= combined_df.ts), 'bolus'] += bolus_value
        
                combined_df.columns = ['ts', 'CGM', 'is_interpolated', 'Basal', 'Bolus']
                combined_df['INS'] = combined_df['Bolus'] + combined_df['Basal']
                combined_df.dropna(subset=['Basal', 'Bolus'], how='any', inplace=True)
        
                daily_counts = combined_df[date_column].dt.floor('d').value_counts()        
                valid_days = daily_counts[daily_counts >= 276].index
                valid_combined_df = filter_by_date(combined_df, valid_days, 'ts')
                valid_combined_df = valid_combined_df.reset_index(drop=True)
                valid_combined_df = valid_combined_df.rename(columns = {'ts':'date'})
                valid_combined_df = valid_combined_df.reset_index()

                sid = filename.split('-')[0]
                valid_combined_df['SID'] = sid
                if sid == '2020_567':
                    print(f"-> Removed P#567(2020 ver.)")
                    continue
            

                new_dir = dir_fp.replace("Raw", "Processed")
                os.makedirs(new_dir, exist_ok=True)
                
                new_fn = filename.split('.')[0] + '.csv'
                new_fp = new_dir + new_fn
                print(f"--> [Saved] New File Name: {new_fp}")
                valid_combined_df.to_csv(new_fp, index=False)

        # Combine to one .pt data
        print('-'*50)
        print(f"Combine to one .pth file in each datasets(train/test)")
        new_dir_train, new_dir_test =  dir_train.replace('Raw', 'Processed'), dir_test.replace('Raw', 'Processed')
        comb_dir_train, comb_dir_test = new_dir_train.replace('Processed', 'Combined'), new_dir_test.replace('Processed', 'Combined')
        os.makedirs(comb_dir_train, exist_ok=True); os.makedirs(comb_dir_test, exist_ok=True);

        # Match SID order between Train & Test Datasets
        train_file_list = os.listdir(new_dir_train)
        test_file_list = os.listdir(new_dir_test)
        
        sid_list = [file.split('-')[0] for file in train_file_list if file.endswith('csv')]
        print(f"Selected Total {len(sid_list)} Patients")

        print(f"[Train -- IOB Transformation & Combine to one .pt file and Save]", end = ' ')
        data_list = []
        for i, sid in enumerate(sid_list):
            file = f'{sid}-ws-training.csv'
            _data = pd.read_csv(new_dir_train+file, index_col=0)
            _data['IOB'] = get_iob(_data)
            _data['SID'] = sid
            data_list.append(_data)
        new_fn = 'cgm+ins+iob.pt'
        new_fp = comb_dir_train + new_fn
        print(f"--> [Saved] New File Name: {comb_dir_train+new_fp}")
        torch.save(data_list, new_fp)

        
        print(f"[Test -- IOB Transformation & Combine to one .pt file and Save]", end = ' ')
        data_list = []
        for i, sid in enumerate(sid_list):
            file = f'{sid}-ws-testing.csv'
            _data = pd.read_csv(new_dir_test+file, index_col=0)
            _data['IOB'] = get_iob(_data)
            _data['SID'] = sid
            data_list.append(_data)
        new_fn = 'cgm+ins+iob.pt'
        new_fp = comb_dir_test + new_fn
        print(f"--> [Saved] New File Name: {comb_dir_test+new_fp}")
        torch.save(data_list, new_fp)
        


    #####################################
    #             DiaTrend              #
    #####################################
    elif args.dataset_name.lower() == 'diatrend':
        # Set the directory where the Excel files are located
        directory =  "Datasets/DiaTrend/Raw/"  # Adjust this to your files' directory
        
        # Names of the sheets we want to manipulate
        cgm_sheet_name = 'CGM'
        basal_sheet_name = 'Basal'
        bolus_sheet_name = 'Bolus'
        
        # We assume the date column in the sheets is named 'Date'
        date_column = 'date'
        rate_column = 'rate'  # Replace with the actual name of the rate column in Basal sheet
        cgm_column = 'mg/dl'
        
        # Process files
        selected_file_list = []
        cnt = 1
        for i, filename in enumerate(os.listdir(directory)):
            if filename.startswith("Subject") and filename.endswith(".xlsx") and (filename!='Subject_test.xlsx'):
                xls = pd.ExcelFile(directory+filename)
                if (len(xls.sheet_names)==3):
                    print(f"[{cnt}] Original File Name: {filename}"); cnt+=1;
                    selected_file_list.append(filename)
                    original_file_path = os.path.join(directory, filename)
                    new_dir = directory.replace('Raw', 'Processed')
                    new_filename = 'Processed_' + filename
                    
                    new_file_path = os.path.join(new_dir, new_filename)
        
                    '''
                    Processing CGM
                    '''
                    # Load the 'CGM' sheet into a pandas DataFrame
                    cgm_df = pd.read_excel(original_file_path, sheet_name=cgm_sheet_name)        
                    # Convert the date column to datetime
                    cgm_df[date_column] = pd.to_datetime(cgm_df[date_column])        
                    # Sort the data based on the date column
                    cgm_df.sort_values(by=date_column, inplace=True)  
        
                    daily_counts = cgm_df[date_column].dt.floor('d').value_counts()        
                    # Filter out days with fewer than 276 entries
                    valid_days = daily_counts[daily_counts >= 276].index     
                    
                    # Round the 'DateTime' column to the nearest 5 minutes
                    cgm_df[date_column] = cgm_df[date_column].dt.round('5min')
                    # Set the date column as the index because resample works on the index
                    cgm_df.set_index(date_column, inplace=True)
                    # Resample the DataFrame to 5-minute intervals
                    cgm_df = cgm_df.resample('5min').first()
                    
                    # if the cgm of 1 days has missings more than 20 minutes, remove that day's data
                    not_missed_cgm_df = remove_consecutive_nans(cgm_df, cgm_column, 4, date_column)
        
                    # indexing interpolated points
                    not_missed_cgm_df['is_interpolated'] = False
                    not_missed_cgm_df.loc[pd.isna(not_missed_cgm_df['mg/dl']), 'is_interpolated'] = True
                    
                    # linear interpolate data which has missings equal or less than 20 minutes
                    interp_cgm_df = not_missed_cgm_df.copy()
                    interp_cgm_df['mg/dl'] = not_missed_cgm_df['mg/dl'].interpolate(method='cubic')
                    cgm_df = interp_cgm_df.drop(columns = ['group', 'nan_count']).reset_index(drop=True)

                    cgm_df.loc[cgm_df['mg/dl']<40, 'mg/dl'] = 40
                    cgm_df.loc[cgm_df['mg/dl']>400, 'mg/dl'] = 400
        
                    '''
                    Processing Insulin
                    '''
                    
                    # Load the 'Basal' and 'Bolus' sheets
                    basal_df = pd.read_excel(original_file_path, sheet_name=basal_sheet_name)
                    bolus_df = pd.read_excel(original_file_path, sheet_name=bolus_sheet_name)
                    
                    # Convert the date columns to datetime
                    basal_df[date_column] = pd.to_datetime(basal_df[date_column])        
                    # Round the 'DateTime' column to the nearest 5 minutes
                    basal_df[date_column] = basal_df[date_column].dt.round('5min')
                    basal_df[rate_column] = basal_df[rate_column].fillna(0)  # Fill NaN values in 'Rate' column with 0
                    # First, ensure your date column is in datetime format and sorted
                    basal_df = basal_df.sort_values(by=date_column)
                    # Set the date column as the index because resample works on the index
                    basal_df.set_index(date_column, inplace=True)
            
                    # Resample the DataFrame to 5-minute intervals
                    basal_df = basal_df.resample('5min').first()
                    basal_df['rate'] = basal_df['rate'].fillna(method='ffill')
                    basal_df['rate'] = basal_df['rate'] / 12
                    basal_df.reset_index(inplace=True)
            
                    ### Bolus
                    bolus_df[date_column] = pd.to_datetime(bolus_df[date_column])
                    bolus_df[date_column] = bolus_df[date_column].dt.round('5min')
                    bolus_df = bolus_df.sort_values(by=date_column)
                    bolus_df.set_index(date_column, inplace=True)
            
                    # Resample the DataFrame to 5-minute intervals
                    bolus_df = bolus_df.resample('5min').first()
                    bolus_df['normal'] = bolus_df['normal'].fillna(0)
                    bolus_df['carbInput'] = bolus_df['carbInput'].fillna(0)
            
                    # Reset the index so that date_column becomes a column again
                    bolus_df.reset_index(inplace=True)
        
                    combined_df = pd.merge(cgm_df, basal_df.loc[:, [date_column, 'rate']], on='date', how='left')
                    combined_df = pd.merge(combined_df, bolus_df.loc[:, [date_column, 'normal', 'carbInput']], on='date', how='left')
                    
                    # Giving the columns appropriate names
                    combined_df.columns = ['date', 'CGM', 'is_interpolated', 'Basal', 'Bolus', 'CHO']
                    combined_df['INS'] = combined_df['Bolus'] + combined_df['Basal']
                    combined_df.dropna(subset=['Basal', 'Bolus'], how='any', inplace=True)
        
                    daily_counts = combined_df[date_column].dt.floor('d').value_counts()        
                    valid_days = daily_counts[daily_counts >= 276].index
                    valid_combined_df = filter_by_date(combined_df, valid_days, date_column)
                    valid_combined_df = valid_combined_df.reset_index(drop=True)
        
                    # Save the modified DataFrames to new sheets in a new file
                    with pd.ExcelWriter(new_file_path, engine='openpyxl') as writer:
                        valid_combined_df.to_excel(writer, sheet_name=cgm_sheet_name, index=False)   
                    print(f"--> [Saved] New File Name: {new_filename}")

        print('-'*50)
        print(f"[IOB Transformation & Combine to one .pt file and Save]", end = ' ')
        file_list = os.listdir(new_dir)
        data_list = []
        for i, file in enumerate(file_list):
            if (os.path.splitext(file)[1]=='.xlsx'):
                _data = pd.read_excel(new_dir+file)
                _data['IOB'] = get_iob(_data)
                _data['SID'] = file.split('.')[0].split('_')[1][-2:]        
                data_list.append(_data)
        print(f"{len(data_list)} Patients")

        final_fp = 'Datasets/DiaTrend/Combined/cgm+ins+iob.pt'
        print(f"Final Processed File Path: {final_fp}")
        torch.save(data_list, final_fp)

    
        

    #####################################
    #           ShanghaiT1DM            #
    #####################################
    elif args.dataset_name.lower() == 'shanghait1dm':
        fp = f'Datasets/ShanghaiT1DM/Raw/'
        new_fp = fp.replace('Raw', 'Processed')
        file_list = os.listdir(fp)
        
        for i, filename in enumerate(file_list):
            if filename.endswith('.csv'):
                data_new = pd.read_csv(fp+filename)
                subject = filename.split('_')[0]+ '_' +filename.split('_')[1]
                print(f"[{i+1}] Original File Name: {filename} ")
            
                data_new['Date'] = pd.to_datetime(data_new['Date'])
                data_new.set_index('Date', inplace=True)
            
                # Resampling to 5-minute intervals
                resampled_data_5min = data_new.resample('5min').first()
                resampled_data_5min['is_interpolated'] = [True if pd.isna(cgm) else False for cgm in resampled_data_5min['CGM (mg / dl)']] 
                
                resampled_data_5min = resampled_data_5min.rename(columns={'CSII - basal insulin (Novolin R, IU / H)': 'Basal'})
                resampled_data_5min = resampled_data_5min.rename(columns={'CSII - bolus insulin (Novolin R, IU)': 'Bolus'})
                resampled_data_5min = resampled_data_5min.rename(columns={'CGM (mg / dl)': 'CGM'})
            
                # Interpolating CGM values
                resampled_data_5min['CGM'] = resampled_data_5min['CGM'].interpolate(method = 'cubic')
                resampled_data_5min.loc[resampled_data_5min['CGM']<40, 'CGM'] = 40
                resampled_data_5min.loc[resampled_data_5min['CGM']>400, 'CGM'] = 400
                
                # Forward filling Basal values as before
                resampled_data_5min['Basal'] = resampled_data_5min['Basal'].fillna(method='ffill')
                
                # Rounding Bolus values to the nearest 5 minutes and filling NaNs with 0
                resampled_data_5min['Bolus'] = resampled_data_5min['Bolus'].fillna(0)
                
                # Convert 'Basal' from units/hour to units/5 minutes
                resampled_data_5min['Basal per 5min'] = resampled_data_5min['Basal'] / 12
                
                # Creating a new column 'INS' that adds 'Bolus' and 'Basal per 5min' values
                resampled_data_5min['INS'] = resampled_data_5min['Bolus'] + resampled_data_5min['Basal per 5min']
                
                # Resetting the index to make 'Date' a column again
                resampled_data_5min.reset_index(inplace=True)
                
                # Define the file path for saving the Excel file
                # New Excel filename using the extracted number
                new_filename = f'{new_fp}5minResamepled_{subject}.xlsx'
                print(f"--> [Saved] New File Name: {new_filename} ")
                resampled_data_5min.to_excel(new_filename, index=False)

        print(f"[IOB Transformation & Combine to one .pt file and Save]", end = ' ')
        file_list = os.listdir(new_fp)                
        all_data = []
        for i, file in enumerate(file_list):
            if (os.path.splitext(file)[1]=='.xlsx'):        
                sid = file.split('.')[0].split('_')[1]+'_'+file.split('.')[0].split('_')[2]
                
                _data = pd.read_excel(new_fp+file)
                data = _data.loc[:, ['Date', 'CGM', 'INS', 'Bolus', 'Basal per 5min', 'is_interpolated']].rename(columns = {'Date': 'date'})
                data['IOB'] = get_iob(data)
                data['SID'] = sid
                all_data.append(data)
        combined_fp = 'Datasets/ShanghaiT1DM/Combined/cgm+ins+iob.pt'
        print(f"{len(all_data)} Patients\nFinal Processed File Path: {combined_fp}")
        torch.save(all_data,combined_fp)
        

if __name__ == "__main__":
    main()
