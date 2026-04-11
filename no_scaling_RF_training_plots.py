#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 12:17:04 2025

@author: eta


With datasets
1) authors dataset + dnabert-pca features
2) my synthetic pair-concat dataset + dnabert-pca features

Do RF model training without scaling
Make predictions, plots
"""

# make plots similar to one in the paper using all rows of training/test dataset, train on chr17, test on chr14
# change labels in plots, color?

# train RF model, save predictions as csv files
# path: complete path to the file that will be saved as predictions, including filename.csv, in string form
def train17_test14_dnabert(file_train, file_test, path):
    # import libraries
    # pip install wheel pandas scikit-learn seaborn joblib scipy
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    
    df_train = pd.read_csv(file_train)
    df_test = pd.read_csv(file_test)    
    
    # Now extract features and target
    # Columns to exclude from X
    columns_to_drop = ['chr', 'start1', 'end1', 'start2', 'end2', 'Count']
    
    # Features
    X_train = df_train.drop(columns=columns_to_drop)
    X_test = df_test.drop(columns=columns_to_drop)
    
    # Target
    y_train = df_train['Count']
    y_test = df_test['Count']
    
    # Transform y from column vector to 1D array
    #y_train_array = y_train.values.ravel()
    y_train_array = y_train.to_numpy().flatten()
    #y_test_array = y_test.values.ravel()
    y_test_array = y_test.to_numpy().flatten()

    # Define the model
    model = RandomForestRegressor(n_estimators=20, max_features=0.3, random_state=42, bootstrap=True)


    model.fit(X_train, y_train_array)
    predictions = pd.DataFrame({
        'start1': df_test['start1'],
        'end1': df_test['end1'],
        'start2': df_test['start2'],
        'end2': df_test['end2'],
        'TrueValue': y_test_array,
        'PredictedValue': model.predict(X_test),
        'Distance': df_test['Distance']
        })


    # save predictions to a file
    predictions.to_csv(path, index=False)

    
    
    
# paths are full file paths to predictions with .csv
# save path is full file path to output .png file
def pearson_plots_train17_test14_dnabert_gm12878(path_dnabert, path_control, save_path):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    gm12878_pc_dnabert = pd.read_csv(path_dnabert)
    gm12878_pc_control = pd.read_csv(path_control)
    
    # >0, <=5000
    # >5000, <=10,000
    # ...
    # >995,000 , <=1,000,000
    # total 200 bins
    
    pearson_array_gm12878_pc_dnabert = np.zeros(200)
    pearson_array_gm12878_pc_control = np.zeros(200)
    
    for i in range(200):
        min_value = i*5000
        max_value = min_value + 5000
        filtered_gm12878_pc_dnabert = gm12878_pc_dnabert[(gm12878_pc_dnabert['Distance'] > min_value) & (gm12878_pc_dnabert['Distance'] <= max_value)]
        filtered_gm12878_pc_control = gm12878_pc_control[(gm12878_pc_control['Distance'] > min_value) & (gm12878_pc_control['Distance'] <= max_value)]

        correlation_gm12878_pc_dnabert = filtered_gm12878_pc_dnabert['TrueValue'].corr(filtered_gm12878_pc_dnabert['PredictedValue'], method='pearson')
        correlation_gm12878_pc_control = filtered_gm12878_pc_control['TrueValue'].corr(filtered_gm12878_pc_control['PredictedValue'], method='pearson')

        pearson_array_gm12878_pc_dnabert[i] = correlation_gm12878_pc_dnabert
        pearson_array_gm12878_pc_control[i] = correlation_gm12878_pc_control

    
    # X-axis: indices (0 to 199)
    x = np.arange(200)
    x = np.array(x) / 200  # Normalize x values to range [0,1]
    # Plot the three arrays
    plt.figure(figsize=(5, 5))  # Set figure size
    plt.plot(x, pearson_array_gm12878_pc_dnabert, label="PAIR-CONCAT (DNABERT)", linestyle="-", marker="o", markersize=3, color="orange")
    plt.plot(x, pearson_array_gm12878_pc_control, label="PAIR-CONCAT (CONTROL)", linestyle="--", marker="s", markersize=3, color="purple")

    
    # Labels and title
    plt.xlabel("Distance (Mb)")
    plt.ylabel("Correlation")
    plt.title("Gm12878 - train chr17, test chr14")
    plt.legend()  # Show legend
    
    # Set axis limits to [0,1]
    plt.xlim(0, 1)
    plt.ylim(-0.2, 0.8)
    
    # Set ticks and grid
    plt.xticks([0, 0.5, 1])
    plt.yticks([-0.2, 0, 0.2, 0.4, 0.6])
    plt.grid(True, linestyle="--", alpha=0.5)
     
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High-resolution save
    
    print(f"Plot saved at: {save_path}")

#####
# paths are full file paths to predictions with .csv
# save path is full file path to output .png file
def pearson_plots_train17_test14_dnabert_K562(path_dnabert, path_control, save_path):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    K562_pc_dnabert = pd.read_csv(path_dnabert)
    K562_pc_control = pd.read_csv(path_control)
    
    # >0, <=5000
    # >5000, <=10,000
    # ...
    # >995,000 , <=1,000,000
    # total 200 bins
    
    pearson_array_K562_pc_dnabert = np.zeros(200)
    pearson_array_K562_pc_control = np.zeros(200)
    
    for i in range(200):
        min_value = i*5000
        max_value = min_value + 5000
        filtered_K562_pc_dnabert = K562_pc_dnabert[(K562_pc_dnabert['Distance'] > min_value) & (K562_pc_dnabert['Distance'] <= max_value)]
        filtered_K562_pc_control = K562_pc_control[(K562_pc_control['Distance'] > min_value) & (K562_pc_control['Distance'] <= max_value)]

        correlation_K562_pc_dnabert = filtered_K562_pc_dnabert['TrueValue'].corr(filtered_K562_pc_dnabert['PredictedValue'], method='pearson')
        correlation_K562_pc_control = filtered_K562_pc_control['TrueValue'].corr(filtered_K562_pc_control['PredictedValue'], method='pearson')

        pearson_array_K562_pc_dnabert[i] = correlation_K562_pc_dnabert
        pearson_array_K562_pc_control[i] = correlation_K562_pc_control

    
    # X-axis: indices (0 to 199)
    x = np.arange(200)
    x = np.array(x) / 200  # Normalize x values to range [0,1]
    # Plot the three arrays
    plt.figure(figsize=(5, 5))  # Set figure size
    plt.plot(x, pearson_array_K562_pc_dnabert, label="PAIR-CONCAT (DNABERT)", linestyle="-", marker="o", markersize=3, color="orange")
    plt.plot(x, pearson_array_K562_pc_control, label="PAIR-CONCAT (CONTROL)", linestyle="--", marker="s", markersize=3, color="purple")

    
    # Labels and title
    plt.xlabel("Distance (Mb)")
    plt.ylabel("Correlation")
    plt.title("K562 - train chr17, test chr14")
    plt.legend()  # Show legend
    
    # Set axis limits to [0,1]
    plt.xlim(0, 1)
    plt.ylim(-0.2, 0.8)
    
    # Set ticks and grid
    plt.xticks([0, 0.5, 1])
    plt.yticks([-0.2, 0, 0.2, 0.4, 0.6])
    plt.grid(True, linestyle="--", alpha=0.5)
     
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High-resolution save
    
    print(f"Plot saved at: {save_path}")
    
    
# paths are full file paths to predictions with .csv
# save path is full file path to output .png file
def pearson_plots_train17_test14_dnabert_Huvec(path_dnabert, path_control, save_path):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    Huvec_pc_dnabert = pd.read_csv(path_dnabert)
    Huvec_pc_control = pd.read_csv(path_control)
    
    # >0, <=5000
    # >5000, <=10,000
    # ...
    # >995,000 , <=1,000,000
    # total 200 bins
    
    pearson_array_Huvec_pc_dnabert = np.zeros(200)
    pearson_array_Huvec_pc_control = np.zeros(200)
    
    for i in range(200):
        min_value = i*5000
        max_value = min_value + 5000
        filtered_Huvec_pc_dnabert = Huvec_pc_dnabert[(Huvec_pc_dnabert['Distance'] > min_value) & (Huvec_pc_dnabert['Distance'] <= max_value)]
        filtered_Huvec_pc_control = Huvec_pc_control[(Huvec_pc_control['Distance'] > min_value) & (Huvec_pc_control['Distance'] <= max_value)]

        correlation_Huvec_pc_dnabert = filtered_Huvec_pc_dnabert['TrueValue'].corr(filtered_Huvec_pc_dnabert['PredictedValue'], method='pearson')
        correlation_Huvec_pc_control = filtered_Huvec_pc_control['TrueValue'].corr(filtered_Huvec_pc_control['PredictedValue'], method='pearson')

        pearson_array_Huvec_pc_dnabert[i] = correlation_Huvec_pc_dnabert
        pearson_array_Huvec_pc_control[i] = correlation_Huvec_pc_control

    
    # X-axis: indices (0 to 199)
    x = np.arange(200)
    x = np.array(x) / 200  # Normalize x values to range [0,1]
    # Plot the three arrays
    plt.figure(figsize=(5, 5))  # Set figure size
    plt.plot(x, pearson_array_Huvec_pc_dnabert, label="PAIR-CONCAT (DNABERT)", linestyle="-", marker="o", markersize=3, color="orange")
    plt.plot(x, pearson_array_Huvec_pc_control, label="PAIR-CONCAT (CONTROL)", linestyle="--", marker="s", markersize=3, color="purple")

    
    # Labels and title
    plt.xlabel("Distance (Mb)")
    plt.ylabel("Correlation")
    plt.title("Huvec - train chr17, test chr14")
    plt.legend()  # Show legend
    
    # Set axis limits to [0,1]
    plt.xlim(0, 1)
    plt.ylim(-0.2, 0.8)
    
    # Set ticks and grid
    plt.xticks([0, 0.5, 1])
    plt.yticks([-0.2, 0, 0.2, 0.4, 0.6])
    plt.grid(True, linestyle="--", alpha=0.5)
     
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High-resolution save
    
    print(f"Plot saved at: {save_path}")
    
    
    
    
    
# paths are full file paths to predictions with .csv
# save path is full file path to output .png file
def pearson_plots_train17_test14_dnabert_Hmec(path_dnabert, path_control, save_path):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    Hmec_pc_dnabert = pd.read_csv(path_dnabert)
    Hmec_pc_control = pd.read_csv(path_control)
    
    # >0, <=5000
    # >5000, <=10,000
    # ...
    # >995,000 , <=1,000,000
    # total 200 bins
    
    pearson_array_Hmec_pc_dnabert = np.zeros(200)
    pearson_array_Hmec_pc_control = np.zeros(200)
    
    for i in range(200):
        min_value = i*5000
        max_value = min_value + 5000
        filtered_Hmec_pc_dnabert = Hmec_pc_dnabert[(Hmec_pc_dnabert['Distance'] > min_value) & (Hmec_pc_dnabert['Distance'] <= max_value)]
        filtered_Hmec_pc_control = Hmec_pc_control[(Hmec_pc_control['Distance'] > min_value) & (Hmec_pc_control['Distance'] <= max_value)]

        correlation_Hmec_pc_dnabert = filtered_Hmec_pc_dnabert['TrueValue'].corr(filtered_Hmec_pc_dnabert['PredictedValue'], method='pearson')
        correlation_Hmec_pc_control = filtered_Hmec_pc_control['TrueValue'].corr(filtered_Hmec_pc_control['PredictedValue'], method='pearson')

        pearson_array_Hmec_pc_dnabert[i] = correlation_Hmec_pc_dnabert
        pearson_array_Hmec_pc_control[i] = correlation_Hmec_pc_control

    
    # X-axis: indices (0 to 199)
    x = np.arange(200)
    x = np.array(x) / 200  # Normalize x values to range [0,1]
    # Plot the three arrays
    plt.figure(figsize=(5, 5))  # Set figure size
    plt.plot(x, pearson_array_Hmec_pc_dnabert, label="PAIR-CONCAT (DNABERT)", linestyle="-", marker="o", markersize=3, color="orange")
    plt.plot(x, pearson_array_Hmec_pc_control, label="PAIR-CONCAT (CONTROL)", linestyle="--", marker="s", markersize=3, color="purple")

    
    # Labels and title
    plt.xlabel("Distance (Mb)")
    plt.ylabel("Correlation")
    plt.title("Hmec - train chr17, test chr14")
    plt.legend()  # Show legend
    
    # Set axis limits to [0,1]
    plt.xlim(0, 1)
    plt.ylim(-0.2, 0.8)
    
    # Set ticks and grid
    plt.xticks([0, 0.5, 1])
    plt.yticks([-0.2, 0, 0.2, 0.4, 0.6])
    plt.grid(True, linestyle="--", alpha=0.5)
     
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High-resolution save
    
    print(f"Plot saved at: {save_path}")
    
    
    
    
# paths are full file paths to predictions with .csv
# save path is full file path to output .png file
def pearson_plots_train17_test14_dnabert_Nhek(path_dnabert, path_control, save_path):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    Nhek_pc_dnabert = pd.read_csv(path_dnabert)
    Nhek_pc_control = pd.read_csv(path_control)
    
    # >0, <=5000
    # >5000, <=10,000
    # ...
    # >995,000 , <=1,000,000
    # total 200 bins
    
    pearson_array_Nhek_pc_dnabert = np.zeros(200)
    pearson_array_Nhek_pc_control = np.zeros(200)
    
    for i in range(200):
        min_value = i*5000
        max_value = min_value + 5000
        filtered_Nhek_pc_dnabert = Nhek_pc_dnabert[(Nhek_pc_dnabert['Distance'] > min_value) & (Nhek_pc_dnabert['Distance'] <= max_value)]
        filtered_Nhek_pc_control = Nhek_pc_control[(Nhek_pc_control['Distance'] > min_value) & (Nhek_pc_control['Distance'] <= max_value)]

        correlation_Nhek_pc_dnabert = filtered_Nhek_pc_dnabert['TrueValue'].corr(filtered_Nhek_pc_dnabert['PredictedValue'], method='pearson')
        correlation_Nhek_pc_control = filtered_Nhek_pc_control['TrueValue'].corr(filtered_Nhek_pc_control['PredictedValue'], method='pearson')

        pearson_array_Nhek_pc_dnabert[i] = correlation_Nhek_pc_dnabert
        pearson_array_Nhek_pc_control[i] = correlation_Nhek_pc_control

    
    # X-axis: indices (0 to 199)
    x = np.arange(200)
    x = np.array(x) / 200  # Normalize x values to range [0,1]
    # Plot the three arrays
    plt.figure(figsize=(5, 5))  # Set figure size
    plt.plot(x, pearson_array_Nhek_pc_dnabert, label="PAIR-CONCAT (DNABERT)", linestyle="-", marker="o", markersize=3, color="orange")
    plt.plot(x, pearson_array_Nhek_pc_control, label="PAIR-CONCAT (CONTROL)", linestyle="--", marker="s", markersize=3, color="purple")

    
    # Labels and title
    plt.xlabel("Distance (Mb)")
    plt.ylabel("Correlation")
    plt.title("Nhek - train chr17, test chr14")
    plt.legend()  # Show legend
    
    # Set axis limits to [0,1]
    plt.xlim(0, 1)
    plt.ylim(-0.2, 0.8)
    
    # Set ticks and grid
    plt.xticks([0, 0.5, 1])
    plt.yticks([-0.2, 0, 0.2, 0.4, 0.6])
    plt.grid(True, linestyle="--", alpha=0.5)
     
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High-resolution save
    
    print(f"Plot saved at: {save_path}")



import os
from concurrent.futures import ProcessPoolExecutor

def run_train(args):
    file1, file2, out_path = args
    train17_test14_dnabert(file1, file2, out_path)

def run_plot(args):
    path_dnabert, path_control, save_path, plot_func = args
    plot_func(path_dnabert, path_control, save_path)




def main():
    # file path setup
    base_path_predictions = '/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/largest_authors_dataset_with_dnabert_features/predictions_plots_no_scaling'
    base_path_predictions_2 = '/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest/predictions_plots_no_scaling'
    # authors dataset
    path1 = os.path.join(base_path_predictions, 'predictions_authors_pc_Gm12878_dnabert_train17_test14.csv')
    path2 = os.path.join(base_path_predictions, 'predictions_authors_pc_K562_dnabert_train17_test14.csv')
    path3 = os.path.join(base_path_predictions, 'predictions_authors_pc_Hmec_dnabert_train17_test14.csv')
    path4 = os.path.join(base_path_predictions, 'predictions_authors_pc_Huvec_dnabert_train17_test14.csv')
    path5 = os.path.join(base_path_predictions, 'predictions_authors_pc_Nhek_dnabert_train17_test14.csv')
    path6 = os.path.join(base_path_predictions, 'predictions_authors_pc_Gm12878_control_train17_test14.csv')
    path7 = os.path.join(base_path_predictions, 'predictions_authors_pc_K562_control_train17_test14.csv')
    path8 = os.path.join(base_path_predictions, 'predictions_authors_pc_Hmec_control_train17_test14.csv')
    path9 = os.path.join(base_path_predictions, 'predictions_authors_pc_Huvec_control_train17_test14.csv')
    path10 = os.path.join(base_path_predictions, 'predictions_authors_pc_Nhek_control_train17_test14.csv')
    # my dataset
    path11 = os.path.join(base_path_predictions_2, 'predictions_mine_pc_Gm12878_dnabert_train17_test14.csv')
    path12 = os.path.join(base_path_predictions_2, 'predictions_mine_pc_K562_dnabert_train17_test14.csv')
    path13 = os.path.join(base_path_predictions_2, 'predictions_mine_pc_Hmec_dnabert_train17_test14.csv')
    path14 = os.path.join(base_path_predictions_2, 'predictions_mine_pc_Huvec_dnabert_train17_test14.csv')
    path15 = os.path.join(base_path_predictions_2, 'predictions_mine_pc_Nhek_dnabert_train17_test14.csv')
    path16 = os.path.join(base_path_predictions_2, 'predictions_mine_pc_Gm12878_control_train17_test14.csv')
    path17 = os.path.join(base_path_predictions_2, 'predictions_mine_pc_K562_control_train17_test14.csv')
    path18 = os.path.join(base_path_predictions_2, 'predictions_mine_pc_Hmec_control_train17_test14.csv')
    path19 = os.path.join(base_path_predictions_2, 'predictions_mine_pc_Huvec_control_train17_test14.csv')
    path20 = os.path.join(base_path_predictions_2, 'predictions_mine_pc_Nhek_control_train17_test14.csv')
    
    # file paths
    file_path_authors_1 = '/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/largest_authors_dataset_with_dnabert_features/with_dnabert_features_Gm12878_chr17_feature.csv'
    file_path_authors_2 = '/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/largest_authors_dataset_with_dnabert_features/with_dnabert_features_Gm12878_chr14_feature.csv'
    file_path_authors_3 = '/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/largest_authors_dataset_with_dnabert_features/with_dnabert_features_K562_chr17_feature.csv'
    file_path_authors_4 = '/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/largest_authors_dataset_with_dnabert_features/with_dnabert_features_K562_chr14_feature.csv'
    file_path_authors_5 = '/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/largest_authors_dataset_with_dnabert_features/with_dnabert_features_Hmec_chr17_feature.csv'
    file_path_authors_6 = '/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/largest_authors_dataset_with_dnabert_features/with_dnabert_features_Hmec_chr14_feature.csv'
    file_path_authors_7 = '/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/largest_authors_dataset_with_dnabert_features/with_dnabert_features_Huvec_chr17_feature.csv'
    file_path_authors_8 = '/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/largest_authors_dataset_with_dnabert_features/with_dnabert_features_Huvec_chr14_feature.csv'
    file_path_authors_9 = '/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/largest_authors_dataset_with_dnabert_features/with_dnabert_features_Nhek_chr17_feature.csv'
    file_path_authors_10 = '/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/largest_authors_dataset_with_dnabert_features/with_dnabert_features_Nhek_chr14_feature.csv'

    file_path_mine_1 = '/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest/with_dnabert_features_Gm12878_chr17_pair_concat.csv'
    file_path_mine_2 = '/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest/with_dnabert_features_Gm12878_chr14_pair_concat.csv'
    file_path_mine_3 = '/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest/with_dnabert_features_K562_chr17_pair_concat.csv'
    file_path_mine_4 = '/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest/with_dnabert_features_K562_chr14_pair_concat.csv'
    file_path_mine_5 = '/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest/with_dnabert_features_Hmec_chr17_pair_concat.csv'
    file_path_mine_6 = '/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest/with_dnabert_features_Hmec_chr14_pair_concat.csv'
    file_path_mine_7 = '/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest/with_dnabert_features_Huvec_chr17_pair_concat.csv'
    file_path_mine_8 = '/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest/with_dnabert_features_Huvec_chr14_pair_concat.csv'
    file_path_mine_9 = '/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest/with_dnabert_features_Nhek_chr17_pair_concat.csv'
    file_path_mine_10 = '/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest/with_dnabert_features_Nhek_chr14_pair_concat.csv'

    
    # --- Prepare training jobs ---
    train_jobs = [
        # authors
        (file_path_authors_1, file_path_authors_2, path1),
        (file_path_authors_3, file_path_authors_4, path2),
        (file_path_authors_5, file_path_authors_6, path3),
        (file_path_authors_7, file_path_authors_8, path4),
        (file_path_authors_9, file_path_authors_10, path5),
        # authors control
        ('/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/txt_to_csv_window/Gm12878_chr17_feature.csv', '/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/txt_to_csv_window/Gm12878_chr14_feature.csv', path6),
        ('/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/txt_to_csv_window/K562_chr17_feature.csv', '/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/txt_to_csv_window/K562_chr14_feature.csv', path7),
        ('/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/txt_to_csv_window/Hmec_chr17_feature.csv', '/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/txt_to_csv_window/Hmec_chr14_feature.csv', path8),
        ('/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/txt_to_csv_window/Huvec_chr17_feature.csv', '/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/txt_to_csv_window/Huvec_chr14_feature.csv', path9),
        ('/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/txt_to_csv_window/Nhek_chr17_feature.csv', '/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/txt_to_csv_window/Nhek_chr14_feature.csv', path10),
        # mine
        (file_path_mine_1, file_path_mine_2, path11),
        (file_path_mine_3, file_path_mine_4, path12),
        (file_path_mine_5, file_path_mine_6, path13),
        (file_path_mine_7, file_path_mine_8, path14),
        (file_path_mine_9, file_path_mine_10, path15),
        # mine control
        ('/sybig/home/eta/Masterthesis/scripts/make_new_datasets_from_scratch/pair_concat_features/pair_concat_features/Gm12878_chr17_pair_concat.csv', '/sybig/home/eta/Masterthesis/scripts/make_new_datasets_from_scratch/pair_concat_features/pair_concat_features/Gm12878_chr14_pair_concat.csv', path16),
        ('/sybig/home/eta/Masterthesis/scripts/make_new_datasets_from_scratch/pair_concat_features/pair_concat_features/K562_chr17_pair_concat.csv', '/sybig/home/eta/Masterthesis/scripts/make_new_datasets_from_scratch/pair_concat_features/pair_concat_features/K562_chr14_pair_concat.csv', path17),
        ('/sybig/home/eta/Masterthesis/scripts/make_new_datasets_from_scratch/pair_concat_features/pair_concat_features/Hmec_chr17_pair_concat.csv', '/sybig/home/eta/Masterthesis/scripts/make_new_datasets_from_scratch/pair_concat_features/pair_concat_features/Hmec_chr14_pair_concat.csv', path18),
        ('/sybig/home/eta/Masterthesis/scripts/make_new_datasets_from_scratch/pair_concat_features/pair_concat_features/Huvec_chr17_pair_concat.csv', '/sybig/home/eta/Masterthesis/scripts/make_new_datasets_from_scratch/pair_concat_features/pair_concat_features/Huvec_chr14_pair_concat.csv', path19),
        ('/sybig/home/eta/Masterthesis/scripts/make_new_datasets_from_scratch/pair_concat_features/pair_concat_features/Nhek_chr17_pair_concat.csv', '/sybig/home/eta/Masterthesis/scripts/make_new_datasets_from_scratch/pair_concat_features/pair_concat_features/Nhek_chr14_pair_concat.csv', path20),
    ]
    
    # --- Prepare plotting jobs ---
    plot_jobs = [
        (path1, path6, os.path.join(base_path_predictions, 'pearson_plot_authors_train17_test14_gm12878_pc_dnabert.png'), pearson_plots_train17_test14_dnabert_gm12878),
        (path2, path7, os.path.join(base_path_predictions, 'pearson_plot_authors_train17_test14_K562_pc_dnabert.png'), pearson_plots_train17_test14_dnabert_K562),
        (path3, path8, os.path.join(base_path_predictions, 'pearson_plot_authors_train17_test14_Hmec_pc_dnabert.png'), pearson_plots_train17_test14_dnabert_Hmec),
        (path4, path9, os.path.join(base_path_predictions, 'pearson_plot_authors_train17_test14_Huvec_pc_dnabert.png'), pearson_plots_train17_test14_dnabert_Huvec),
        (path5, path10, os.path.join(base_path_predictions, 'pearson_plot_authors_train17_test14_Nhek_pc_dnabert.png'), pearson_plots_train17_test14_dnabert_Nhek),
        (path11, path16, os.path.join(base_path_predictions_2, 'pearson_plot_mine_train17_test14_gm12878_pc_dnabert.png'), pearson_plots_train17_test14_dnabert_gm12878),
        (path12, path17, os.path.join(base_path_predictions_2, 'pearson_plot_mine_train17_test14_K562_pc_dnabert.png'), pearson_plots_train17_test14_dnabert_K562),
        (path13, path18, os.path.join(base_path_predictions_2, 'pearson_plot_mine_train17_test14_Hmec_pc_dnabert.png'), pearson_plots_train17_test14_dnabert_Hmec),
        (path14, path19, os.path.join(base_path_predictions_2, 'pearson_plot_mine_train17_test14_Huvec_pc_dnabert.png'), pearson_plots_train17_test14_dnabert_Huvec),
        (path15, path20, os.path.join(base_path_predictions_2, 'pearson_plot_mine_train17_test14_Nhek_pc_dnabert.png'), pearson_plots_train17_test14_dnabert_Nhek),
    ]
    
    # --- Parallel execution ---
    with ProcessPoolExecutor(max_workers=10) as executor:
        executor.map(run_train, train_jobs)
    with ProcessPoolExecutor(max_workers=10) as executor:
        executor.map(run_plot, plot_jobs)

if __name__ == '__main__':
    main()