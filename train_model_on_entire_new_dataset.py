#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 11:41:37 2025

@author: eta

For Gm12878, all chr, train one RF model on all data on 4 cell lines and 23 chr (sample 30,000 rows for each file)
test on each chr of Gm12878, make plots
=> repeat for 5 cell lines
"""
import pandas as pd
import os
from pathlib import Path
import re



def aggregate_data(file_paths, output_path, sample_size=30000):
    """
    Sample up to sample_size rows randomly from each CSV file,
    verify column consistency, and append them vertically
    into a single CSV file.

    Parameters:
    - file_paths: list of str, paths to input CSV files.
    - output_path: str, path to save the merged CSV.
    - sample_size: int, number of rows to sample from each file.
    """
    reference_columns = None
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    for i, path in enumerate(file_paths):
        df = pd.read_csv(path)
        
        if reference_columns is None:
            reference_columns = df.columns.tolist()
        else:
            if df.columns.tolist() != reference_columns:
                raise ValueError(f"File {path} has different columns!")
        
        # Sample rows (or all if less than sample_size)
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        # Append to CSV: write header only for the first file
        if i == 0:
            sample_df.to_csv(output_path, index=False, mode='w')
        else:
            sample_df.to_csv(output_path, index=False, mode='a', header=False)
    
    print(f"Saved sampled merged file to {output_path}")




def get_matching_files(directory, substrings):
    """
    Get files in a directory (non-recursive) that contain any of the given substrings in their names.

    Parameters:
    - directory: str or Path
    - substrings: str or list of str

    Returns:
    - List of Path objects
    """
    directory = Path(directory)
    if isinstance(substrings, str):
        substrings = [substrings]

    return [
        file for file in directory.iterdir()
        if file.is_file() and any(s in file.name for s in substrings)
    ]



# train RF model, save predictions as csv files
# make plots of feature importances, save them
# path: complete path to the file that will be saved as predictions, including filename.csv, in string form
def train_and_save_model(file_train, save_path, n_jobs):  # save_path = .joblib file path
    # import libraries
    # pip install wheel pandas scikit-learn seaborn joblib scipy
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from joblib import dump, load
    
    df_train = pd.read_csv(file_train)
        
    df_train_filtered = df_train
    
    # Now extract features and target
    # Columns to exclude from X
    columns_to_drop = ['chr', 'start1', 'end1', 'start2', 'end2', 'Count']
    
    # Features
    X_train = df_train_filtered.drop(columns=columns_to_drop)
    
    # Target
    y_train = df_train_filtered['Count']

    # scale and normalize columns in X, since Distance column has very high values compared to other columns
    # Initialize the scaler
    scaler = StandardScaler()
    scaler.fit(X_train)  # Fit on train set only

    # Transform train and test sets
    X_train_scaled = scaler.transform(X_train)
    
    # Transform y from column vector to 1D array
    y_train_array = y_train.to_numpy().flatten()

    # Define the model
    model = RandomForestRegressor(n_estimators=20, max_features=0.3, random_state=42, bootstrap=True, n_jobs=n_jobs)


    model.fit(X_train_scaled, y_train_array)
    
    # save model
    dump(model, save_path)  # .joblib file path


    
# train RF model, save predictions as csv files
# make plots of feature importances, save them
# path: complete path to the file that will be saved as predictions, including filename.csv, in string form
def test_model_predictions(file_train, file_test, model_path, save_path):
    # import libraries
    # pip install wheel pandas scikit-learn seaborn joblib scipy
    from sklearn.preprocessing import StandardScaler
    from joblib import dump, load
    
    model = load(model_path)
    
    df_train = pd.read_csv(file_train)
    df_test = pd.read_csv(file_test)    
        
    # Randomly sample 30,000 rows (or fewer) from df_train and df_test
    df_train_filtered = df_train
    df_test_filtered = df_test
    
    # Now extract features and target
    # Columns to exclude from X
    columns_to_drop = ['chr', 'start1', 'end1', 'start2', 'end2', 'Count']
    
    # Features
    X_train = df_train_filtered.drop(columns=columns_to_drop)
    X_test = df_test_filtered.drop(columns=columns_to_drop)
    
    # Target
    y_train = df_train_filtered['Count']
    y_test = df_test_filtered['Count']


    # scale and normalize columns in X, since Distance column has very high values compared to other columns
    # Initialize the scaler
    scaler = StandardScaler()
    scaler.fit(X_train)  # Fit on train set only

    # Transform train and test sets
    # Yes — you must scale the test set using the same scaler fitted on the training set.
    #Why?
    #The model was trained on features transformed (scaled) using the training data.
    #To ensure consistent input representation, the test set must be scaled with the same parameters (e.g., mean and std) as the training set.
    #Never fit a new scaler on the test set, as that introduces data leakage.
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Do not fit again!
    
    # Transform y from column vector to 1D array
    #y_train_array = y_train.values.ravel()
    y_train_array = y_train.to_numpy().flatten()
    #y_test_array = y_test.values.ravel()
    y_test_array = y_test.to_numpy().flatten()


    predictions = pd.DataFrame({
        'start1': df_test_filtered['start1'],
        'end1': df_test_filtered['end1'],
        'start2': df_test_filtered['start2'],
        'end2': df_test_filtered['end2'],
        'TrueValue': y_test_array,
        'PredictedValue': model.predict(X_test_scaled),
        'Distance': df_test_filtered['Distance']
        })


    # save predictions to a file
    predictions.to_csv(save_path, index=False)

    
# generate 1 plot that plots 1) predictions of dnabert and 2) predictions of control
def pearson_plot(dnabert_predictions_path, control_predictions_path, save_path):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # Normalize x-axis (in Mb units from 0 to 1)
    x = np.arange(200) / 200

    # Load predictions
    dnabert = pd.read_csv(dnabert_predictions_path)
    control = pd.read_csv(control_predictions_path)

    # Initialize arrays
    pearson_array_dnabert = np.zeros(200)
    pearson_array_control = np.zeros(200)

    # Compute Pearson correlations for each distance bin
    for i in range(200):
        min_value = i * 5000
        max_value = min_value + 5000

        filtered_dnabert = dnabert[(dnabert['Distance'] > min_value) & (dnabert['Distance'] <= max_value)]
        filtered_control = control[(control['Distance'] > min_value) & (control['Distance'] <= max_value)]

        pearson_array_dnabert[i] = filtered_dnabert['TrueValue'].corr(filtered_dnabert['PredictedValue'])
        pearson_array_control[i] = filtered_control['TrueValue'].corr(filtered_control['PredictedValue'])

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot lines
    #ax.plot(x, pearson_array_dnabert, color="pink", label="DNABERT", linewidth=2.5)
    #ax.plot(x, pearson_array_control, color="purple", label="Control", linewidth=2.5)
    # changed colors for better visibility - coral pink and rich purple
    ax.plot(x, pearson_array_dnabert, color="#FF6F61", label="DNABERT", linewidth=2.5)
    ax.plot(x, pearson_array_control, color="#6B5B95", label="Control", linewidth=2.5)


    # Set axis limits and labels
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, 0.8)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6])
    ax.grid(True, linestyle="--", alpha=0.5)

    ax.set_xlabel("Distance (Mb)", fontsize=14)
    ax.set_ylabel("Correlation", fontsize=14)
    
    # Tick label font sizes
    ax.tick_params(axis='both', labelsize=12)

    # Legend with larger font
    ax.legend(loc="upper right", fontsize=12)

    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {save_path}")

    
    
def main():
    # 1. Append dataframes into large dataframes spanning cell lines and chromosomes
    
    # Find all files in directory with any of the four cell lines in the filename
    # convert Path object to string for each file path
    # get_matching_files(directory, substrings)
    file_list_1 = [str(f) for f in get_matching_files('/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest', ['Huvec', 'Hmec', 'K562', 'Nhek'])]
    file_list_2 = [str(f) for f in get_matching_files('/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest', ['Gm12878', 'Hmec', 'K562', 'Nhek'])]
    file_list_3 = [str(f) for f in get_matching_files('/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest', ['Huvec', 'Gm12878', 'K562', 'Nhek'])]
    file_list_4 = [str(f) for f in get_matching_files('/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest', ['Huvec', 'Hmec', 'Gm12878', 'Nhek'])]
    file_list_5 = [str(f) for f in get_matching_files('/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest', ['Huvec', 'Hmec', 'K562', 'Gm12878'])]
    file_list_6 = [str(f) for f in get_matching_files('/sybig/home/eta/Masterthesis/scripts/make_new_datasets_from_scratch/pair_concat_features/pair_concat_features', ['Huvec', 'Hmec', 'K562', 'Nhek'])]
    file_list_7 = [str(f) for f in get_matching_files('/sybig/home/eta/Masterthesis/scripts/make_new_datasets_from_scratch/pair_concat_features/pair_concat_features', ['Gm12878', 'Hmec', 'K562', 'Nhek'])]
    file_list_8 = [str(f) for f in get_matching_files('/sybig/home/eta/Masterthesis/scripts/make_new_datasets_from_scratch/pair_concat_features/pair_concat_features', ['Huvec', 'Gm12878', 'K562', 'Nhek'])]
    file_list_9 = [str(f) for f in get_matching_files('/sybig/home/eta/Masterthesis/scripts/make_new_datasets_from_scratch/pair_concat_features/pair_concat_features', ['Huvec', 'Hmec', 'Gm12878', 'Nhek'])]
    file_list_10 = [str(f) for f in get_matching_files('/sybig/home/eta/Masterthesis/scripts/make_new_datasets_from_scratch/pair_concat_features/pair_concat_features', ['Huvec', 'Hmec', 'K562', 'Gm12878'])]
    
    output_file_1 = "/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/new_data_with_dnabert/test_on_Gm12878/merged_dnabert_without_Gm12878.csv"
    output_file_2 = "/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/new_data_with_dnabert/test_on_Huvec/merged_dnabert_without_Huvec.csv"
    output_file_3 = "/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/new_data_with_dnabert/test_on_Hmec/merged_dnabert_without_Hmec.csv"
    output_file_4 = "/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/new_data_with_dnabert/test_on_K562/merged_dnabert_without_K562.csv"
    output_file_5 = "/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/new_data_with_dnabert/test_on_Nhek/merged_dnabert_without_Nhek.csv"
    output_file_6 = "/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/new_data_without_dnabert/test_on_Gm12878/merged_control_without_Gm12878.csv"
    output_file_7 = "/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/new_data_without_dnabert/test_on_Huvec/merged_control_without_Huvec.csv"
    output_file_8 = "/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/new_data_without_dnabert/test_on_Hmec/merged_control_without_Hmec.csv"
    output_file_9 = "/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/new_data_without_dnabert/test_on_K562/merged_control_without_K562.csv"
    output_file_10 = "/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/new_data_without_dnabert/test_on_Nhek/merged_control_without_Nhek.csv"
    
    # aggregate_data(file_paths, output_path, sample_size=30000)
    # don't need to pass sample size parameter!
    aggregate_data(file_list_1, output_file_1)
    aggregate_data(file_list_2, output_file_2)
    aggregate_data(file_list_3, output_file_3)
    aggregate_data(file_list_4, output_file_4)
    aggregate_data(file_list_5, output_file_5)
    aggregate_data(file_list_6, output_file_6)
    aggregate_data(file_list_7, output_file_7)
    aggregate_data(file_list_8, output_file_8)
    aggregate_data(file_list_9, output_file_9)
    aggregate_data(file_list_10, output_file_10)
    
    # 2. Train RF models on those dataframes and save the models
    model_path_1 = "/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/new_data_with_dnabert/test_on_Gm12878/model_dnabert_without_Gm12878.joblib"
    model_path_2 = "/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/new_data_with_dnabert/test_on_Huvec/model_dnabert_without_Huvec.joblib"
    model_path_3 = "/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/new_data_with_dnabert/test_on_Hmec/model_dnabert_without_Hmec.joblib"
    model_path_4 = "/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/new_data_with_dnabert/test_on_K562/model_dnabert_without_K562.joblib"
    model_path_5 = "/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/new_data_with_dnabert/test_on_Nhek/model_dnabert_without_Nhek.joblib"
    model_path_6 = "/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/new_data_without_dnabert/test_on_Gm12878/model_control_without_Gm12878.joblib"
    model_path_7 = "/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/new_data_without_dnabert/test_on_Huvec/model_control_without_Huvec.joblib"
    model_path_8 = "/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/new_data_without_dnabert/test_on_Hmec/model_control_without_Hmec.joblib"
    model_path_9 = "/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/new_data_without_dnabert/test_on_K562/model_control_without_K562.joblib"
    model_path_10 = "/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/new_data_without_dnabert/test_on_Nhek/model_control_without_Nhek.joblib"
    
    n_jobs = 20    # change parameter - parallel processing
    
    # train_and_save_model(file_train, save_path, n_jobs):  # save_path = .joblib file path
    train_and_save_model(output_file_1, model_path_1, n_jobs)
    train_and_save_model(output_file_2, model_path_2, n_jobs)
    train_and_save_model(output_file_3, model_path_3, n_jobs)
    train_and_save_model(output_file_4, model_path_4, n_jobs)
    train_and_save_model(output_file_5, model_path_5, n_jobs)
    train_and_save_model(output_file_6, model_path_6, n_jobs)
    train_and_save_model(output_file_7, model_path_7, n_jobs)
    train_and_save_model(output_file_8, model_path_8, n_jobs)
    train_and_save_model(output_file_9, model_path_9, n_jobs)
    train_and_save_model(output_file_10, model_path_10, n_jobs)
    
    # 3. Test on remaining data (cell line/chr) and generate prediction files
    # test_model_predictions(file_train, file_test, model_path, save_path)
    # for each training file/model, there are 23 prediction files (for each chromosome, for 1 cell line)
    
    # dnabert - with dnabert columns ('largest')
    for i in [str(i) for i in range(1, 23)] + ['X']: # chromosome numbers in string
        file_test_1 = f'/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest/with_dnabert_features_Gm12878_chr{i}_pair_concat.csv'
        save_path_1 = f'/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/new_data_with_dnabert/test_on_Gm12878/predictions_dnabert_Gm12878_chr{i}.csv'
        file_test_2 = f'/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest/with_dnabert_features_Huvec_chr{i}_pair_concat.csv'
        save_path_2 = f'/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/new_data_with_dnabert/test_on_Huvec/predictions_dnabert_Huvec_chr{i}.csv'
        file_test_3 = f'/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest/with_dnabert_features_Hmec_chr{i}_pair_concat.csv'
        save_path_3 = f'/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/new_data_with_dnabert/test_on_Hmec/predictions_dnabert_Hmec_chr{i}.csv'
        file_test_4 = f'/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest/with_dnabert_features_K562_chr{i}_pair_concat.csv'
        save_path_4 = f'/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/new_data_with_dnabert/test_on_K562/predictions_dnabert_K562_chr{i}.csv'
        file_test_5 = f'/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest/with_dnabert_features_Nhek_chr{i}_pair_concat.csv'
        save_path_5 = f'/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/new_data_with_dnabert/test_on_Nhek/predictions_dnabert_Nhek_chr{i}.csv'
        test_model_predictions(output_file_1, file_test_1, model_path_1, save_path_1)
        test_model_predictions(output_file_2, file_test_2, model_path_2, save_path_2)
        test_model_predictions(output_file_3, file_test_3, model_path_3, save_path_3)
        test_model_predictions(output_file_4, file_test_4, model_path_4, save_path_4)
        test_model_predictions(output_file_5, file_test_5, model_path_5, save_path_5)
    
    # control - pair-concat files (without dnabert columns)
    for i in [str(i) for i in range(1, 23)] + ['X']:
        file_test_6 = f'/sybig/home/eta/Masterthesis/scripts/make_new_datasets_from_scratch/pair_concat_features/pair_concat_features/Gm12878_chr{i}_pair_concat.csv'
        save_path_6 = f'/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/new_data_without_dnabert/test_on_Gm12878/predictions_control_Gm12878_chr{i}.csv'
        file_test_7 = f'/sybig/home/eta/Masterthesis/scripts/make_new_datasets_from_scratch/pair_concat_features/pair_concat_features/Huvec_chr{i}_pair_concat.csv'
        save_path_7 = f'/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/new_data_without_dnabert/test_on_Huvec/predictions_control_Huvec_chr{i}.csv'
        file_test_8 = f'/sybig/home/eta/Masterthesis/scripts/make_new_datasets_from_scratch/pair_concat_features/pair_concat_features/Hmec_chr{i}_pair_concat.csv'
        save_path_8 = f'/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/new_data_without_dnabert/test_on_Hmec/predictions_control_Hmec_chr{i}.csv'
        file_test_9 = f'/sybig/home/eta/Masterthesis/scripts/make_new_datasets_from_scratch/pair_concat_features/pair_concat_features/K562_chr{i}_pair_concat.csv'
        save_path_9 = f'/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/new_data_without_dnabert/test_on_K562/predictions_control_K562_chr{i}.csv'
        file_test_10 = f'/sybig/home/eta/Masterthesis/scripts/make_new_datasets_from_scratch/pair_concat_features/pair_concat_features/Nhek_chr{i}_pair_concat.csv'
        save_path_10 = f'/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/new_data_without_dnabert/test_on_Nhek/predictions_control_Nhek_chr{i}.csv'
        test_model_predictions(output_file_6, file_test_6, model_path_6, save_path_6)
        test_model_predictions(output_file_7, file_test_7, model_path_7, save_path_7)
        test_model_predictions(output_file_8, file_test_8, model_path_8, save_path_8)
        test_model_predictions(output_file_9, file_test_9, model_path_9, save_path_9)
        test_model_predictions(output_file_10, file_test_10, model_path_10, save_path_10)
    
    
    # 4. Plotting in distance-stratified pearson's correlation plots
    # pearson_plot(dnabert_predictions_path, control_predictions_path, save_path)
    # 5*23 plots
    # Define the root directory to search
    root_dir = Path("/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset")
    
    # Regular expression pattern
    pattern = re.compile(r"^predictions_[^_]+_[^_]+_chr[^_]+\.csv$")
    
    # List to store matching file paths
    matching_files = []
    
    # Search all files recursively and match with the pattern
    for file in root_dir.rglob("*.csv"):
        if file.is_file() and pattern.match(file.name):
            matching_files.append(file)
    # convert Path object to string
    matching_files_str = [str(f) for f in matching_files]  # the entire list of paths of predictions files

    for i in [str(i) for i in range(1, 23)] + ['X']:   # for each chromosome, for each cell line, 1 plot and 2 prediction files needed 
        dnabert_predictions_Gm12878 = next(f for f in matching_files_str if 'Gm12878' in f and 'dnabert' in f and i in f)
        dnabert_predictions_Huvec = next(f for f in matching_files_str if 'Huvec' in f and 'dnabert' in f and i in f)
        dnabert_predictions_Hmec = next(f for f in matching_files_str if 'Hmec' in f and 'dnabert' in f and i in f)
        dnabert_predictions_K562 = next(f for f in matching_files_str if 'K562' in f and 'dnabert' in f and i in f)
        dnabert_predictions_Nhek = next(f for f in matching_files_str if 'Nhek' in f and 'dnabert' in f and i in f)
        
        control_predictions_Gm12878 = next(f for f in matching_files_str if 'Gm12878' in f and 'control' in f and i in f)
        control_predictions_Huvec = next(f for f in matching_files_str if 'Huvec' in f and 'control' in f and i in f)
        control_predictions_Hmec = next(f for f in matching_files_str if 'Hmec' in f and 'control' in f and i in f)
        control_predictions_K562 = next(f for f in matching_files_str if 'K562' in f and 'control' in f and i in f)
        control_predictions_Nhek = next(f for f in matching_files_str if 'Nhek' in f and 'control' in f and i in f)
        
        save_plots_path_Gm12878 = f'/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/Plots/pearson_plot_test_on_Gm12878_chr{i}.png'
        save_plots_path_Huvec = f'/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/Plots/pearson_plot_test_on_Huvec_chr{i}.png'
        save_plots_path_Hmec = f'/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/Plots/pearson_plot_test_on_Hmec_chr{i}.png'
        save_plots_path_K562 = f'/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/Plots/pearson_plot_test_on_K562_chr{i}.png'
        save_plots_path_Nhek = f'/sybig/home/eta/Masterthesis/scripts/dataset_analysis/train_model_on_entire_new_dataset/Plots/pearson_plot_test_on_Nhek_chr{i}.png'
        
        pearson_plot(dnabert_predictions_Gm12878, control_predictions_Gm12878, save_plots_path_Gm12878)
        pearson_plot(dnabert_predictions_Huvec, control_predictions_Huvec, save_plots_path_Huvec)
        pearson_plot(dnabert_predictions_Hmec, control_predictions_Hmec, save_plots_path_Hmec)
        pearson_plot(dnabert_predictions_K562, control_predictions_K562, save_plots_path_K562)
        pearson_plot(dnabert_predictions_Nhek, control_predictions_Nhek, save_plots_path_Nhek)

if __name__ == '__main__':
    main()