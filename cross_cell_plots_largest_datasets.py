#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 13:09:09 2025

@author: eta

Make cross-cell line analysis and plots with largest datasets created
"""
from multiprocessing import Pool
import pandas as pd
import numpy as np


# train RF model, save predictions as csv files
# make plots of feature importances, save them
# path: complete path to the file that will be saved as predictions, including filename.csv, in string form
def cross_cell_dnabert(file_train, file_test, path):
    # import libraries
    # pip install wheel pandas scikit-learn seaborn joblib scipy
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler


    #from scipy.integrate import trapz
    
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
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Do not fit again!
    
    # Transform y from column vector to 1D array
    #y_train_array = y_train.values.ravel()
    y_train_array = y_train.to_numpy().flatten()
    #y_test_array = y_test.values.ravel()
    y_test_array = y_test.to_numpy().flatten()

    # Define the model
    model = RandomForestRegressor(n_estimators=20, max_features=0.3, random_state=42, bootstrap=True, n_jobs=1)


    model.fit(X_train_scaled, y_train_array)
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
    predictions.to_csv(path, index=False)
    


    
# generate 1 plot, made of 5*5 subplots
# each subplot plots 1) predictions of dnabert and 2) predictions of control
def pearson_plots_cross_cell(dict1):
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches


    # order given in plot in paper
    order = ['Huvec', 'Hmec', 'Nhek', 'Gm12878', 'K562']

    # Create a figure with 5*5 subplots
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
    
    # X-axis: indices (0 to 199)
    x = np.arange(200)
    x = np.array(x) / 200  # Normalize x values to range [0,1]
    
    for column in range(5):  # index of training cell
        for row in range(5):   # index of test cell
            if row == column:
                continue
            #cross_cell_dnabert = pd.read_csv(f'/sybig/home/eta/Masterthesis/scripts/dataset_analysis/predictions_cell/predictions_dnabert_train_{order[column]}_test_{order[row]}.csv')
            #cross_cell_control = pd.read_csv(f'/sybig/home/eta/Masterthesis/scripts/dataset_analysis/predictions_cell/predictions_control_train_{order[column]}_test_{order[row]}.csv')
            cross_cell_dnabert = pd.read_csv(dict1['dnabert'][order[column]][order[row]])
            cross_cell_control = pd.read_csv(dict1['control'][order[column]][order[row]])

            pearson_array_dnabert = np.zeros(200)
            pearson_array_control = np.zeros(200)
            
            # >0, <=5000
            # >5000, <=10,000
            # ...
            # >995,000 , <=1,000,000
            # total 200 bins
            for i in range(200):
                min_value = i*5000
                max_value = min_value + 5000
                filtered_dnabert = cross_cell_dnabert[(cross_cell_dnabert['Distance'] > min_value) & (cross_cell_dnabert['Distance'] <= max_value)]
                filtered_control = cross_cell_control[(cross_cell_control['Distance'] > min_value) & (cross_cell_control['Distance'] <= max_value)]
                correlation_dnabert = filtered_dnabert['TrueValue'].corr(filtered_dnabert['PredictedValue'], method='pearson')
                correlation_control = filtered_control['TrueValue'].corr(filtered_control['PredictedValue'], method='pearson')
                pearson_array_dnabert[i] = correlation_dnabert
                pearson_array_control[i] = correlation_control

            # Plot on each subplot
            # Plot the three arrays
            axes[row][column].plot(x, pearson_array_dnabert, label="DNABERT", color="pink")
            axes[row][column].plot(x, pearson_array_control, label="CONTROL", color="brown")

            axes[row][column].set_xlim(0, 1)
            axes[row][column].set_ylim(-0.2, 0.8)
            axes[row][column].set_xticks([0, 0.5, 1])
            axes[row][column].set_yticks([-0.2, 0, 0.2, 0.4, 0.6])
            axes[row][column].grid(True, linestyle="--", alpha=0.5)



    # Column labels at the top, row labels at the left
    for i, label in enumerate(order):
        fig.text(0.16 + i * 0.18, 0.93, label, fontsize=10, ha="center", va="top")  # Column labels
        fig.text(0.01, 0.82 - i * 0.16, label, fontsize=10, ha="left", va="center", rotation=90)  # Row labels

    fig.supxlabel("Distance (Mb)", fontsize=10)
    fig.supylabel("Test cell (Correlation)", fontsize=10, x=0)
    fig.text(0.02, 0.98, "Training cell", fontsize=10, ha="left", va="top")


    # Create custom legend handles
    pink_patch = mpatches.Patch(color='pink', label='DNABERT')
    brown_patch = mpatches.Patch(color='brown', label='Control')
    
    # Add the custom legend to the figure
    fig.legend(
        handles=[pink_patch, brown_patch],
        loc="upper right",      # Top right of entire figure
        fontsize=10,
        bbox_to_anchor=(1, 1),  # Adjust if needed
    )
    
    # Adjust layout so legend doesn't overlap
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    
    save_path = dict1['save']
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High-resolution save

    print(f"Plot saved at: {save_path}")
    
    
    

    
##################################################
### procedure
# 1. generate prediction files
# 2. make plots on authors data
# 3. make plots on my data
# 4. make plots on more chromosomes

### syntax
#cross_cell_dnabert(file_train, file_test, path)
#pearson_plots_cross_cell_authors_data()
#pearson_plots_cross_cell_my_data()

# save folder
# '/sybig/home/eta/Masterthesis/scripts/dataset_analysis/cross_cell_predictions_plots_largest_data_authors_mine'

# 1.
# prediction files authors (chr17, 14)
# '/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/txt_to_csv_window'
# prediction files authors-augment (chr17, 14)
# '/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/largest_authors_dataset_with_dnabert_features'
# prediction files mine pair-concat (all chr)
# '/sybig/home/eta/Masterthesis/scripts/make_new_datasets_from_scratch/pair_concat_features/pair_concat_features'
# prediction files mine-dnabert-pca (all chr)
# '/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest'

# 2. make plots

##################################################

# 1. make predictions files and save it in specified directory
cells = ['Huvec', 'Hmec', 'Nhek', 'Gm12878', 'K562']
types = ["dnabert", "control"]
chr_small = ['14', '17']
chr_large = [str(i) for i in range(1,23)] + ['X']

tasks = []

for chrom in chr_small:
    for i in range(len(cells)):
        for j in range(len(cells)):
            if i==j:
                continue
            train_cell = cells[i]
            test_cell = cells[j]

            agg_dir = '/scratch/eta/20_june_analysis/authors_dataset'
            pred_dir = "/scratch/eta/20_june_analysis/predictions_only_authors"

            agg_train = f"{agg_dir}/{train_cell}_chr{chrom}_feature.csv"
            agg_test = f"{agg_dir}/{test_cell}_chr{chrom}_feature.csv"
            pred_path = f"{pred_dir}/predictions_only_authors_chr{chrom}_train_{train_cell}_test_{test_cell}.csv"

            tasks.append((cross_cell_dnabert, agg_train, agg_test, pred_path))

for chrom in chr_small:
    for i in range(len(cells)):
        for j in range(len(cells)):
            if i==j:
                continue
            train_cell = cells[i]
            test_cell = cells[j]

            agg_dir = '/scratch/eta/20_june_analysis/largest_authors_dataset_with_dnabert_features'
            pred_dir = "/scratch/eta/20_june_analysis/predictions_authors_augmented"

            agg_train = f"{agg_dir}/with_dnabert_features_{train_cell}_chr{chrom}_feature.csv"
            agg_test = f"{agg_dir}/with_dnabert_features_{test_cell}_chr{chrom}_feature.csv"
            pred_path = f"{pred_dir}/predictions_authors_augmented_chr{chrom}_train_{train_cell}_test_{test_cell}.csv"

            tasks.append((cross_cell_dnabert, agg_train, agg_test, pred_path))

for chrom in chr_large:
    for i in range(len(cells)):
        for j in range(len(cells)):
            if i==j:
                continue
            train_cell = cells[i]
            test_cell = cells[j]

            agg_dir = '/scratch/eta/20_june_analysis/pair_concat_features'
            pred_dir = "/scratch/eta/20_june_analysis/predictions_mine_without_dnabert"

            agg_train = f"{agg_dir}/{train_cell}_chr{chrom}_pair_concat.csv"
            agg_test = f"{agg_dir}/{test_cell}_chr{chrom}_pair_concat.csv"
            pred_path = f"{pred_dir}/predictions_mine_without_dnabert_chr{chrom}_train_{train_cell}_test_{test_cell}.csv"

            tasks.append((cross_cell_dnabert, agg_train, agg_test, pred_path))
            
for chrom in chr_large:
    for i in range(len(cells)):
        for j in range(len(cells)):
            if i==j:
                continue
            train_cell = cells[i]
            test_cell = cells[j]

            agg_dir = '/scratch/eta/20_june_analysis/dnabert_datasets_largest'
            pred_dir = "/scratch/eta/20_june_analysis/predictions_mine_with_dnabert"

            agg_train = f"{agg_dir}/with_dnabert_features_{train_cell}_chr{chrom}_pair_concat.csv"
            agg_test = f"{agg_dir}/with_dnabert_features_{test_cell}_chr{chrom}_pair_concat.csv"
            pred_path = f"{pred_dir}/predictions_mine_with_dnabert_chr{chrom}_train_{train_cell}_test_{test_cell}.csv"

            tasks.append((cross_cell_dnabert, agg_train, agg_test, pred_path))


# Wrapper function to execute tasks
def run_task(task):
    func, *args = task
    func(*args)

order = ['Huvec', 'Hmec', 'Nhek', 'Gm12878', 'K562']
'''
dict_1 = {
    'dnabert': 
        {
            'Huvec': {
                'Huvec':
                'Hmec':
                'Nhek':
                'Gm12878':
                'K562':
                }
            'Hmec'; {
                }
            'Nhek'; {
                }
            'Gm12878': {
                }
            'K562': {
                }
            }
    'control': 
        {
            'Huvec': {
                }
            'Hmec'; {
                }
            'Nhek'; {
                }
            'Gm12878': {
                }
            'K562': {
                }
            }
    'save': (save_path)
    }
'''
# initialize empty dictionaries
authors_chr14 = {}
authors_chr17 = {}

mine_chr1 = {}
mine_chr2 = {}
mine_chr3 = {}
mine_chr4 = {}
mine_chr5 = {}
mine_chr6 = {}
mine_chr7 = {}
mine_chr8 = {}
mine_chr9 = {}
mine_chr10 = {}
mine_chr11 = {}
mine_chr12 = {}
mine_chr13 = {}
mine_chr14 = {}
mine_chr15 = {}
mine_chr16 = {}
mine_chr17 = {}
mine_chr18 = {}
mine_chr19 = {}
mine_chr20 = {}
mine_chr21 = {}
mine_chr22 = {}
mine_chrX = {}


# make list of dictionaries
list_of_dictionaries_authors = [authors_chr14, authors_chr17]
list_of_dictionaries_mine = [mine_chr1, mine_chr2, mine_chr3, mine_chr4, mine_chr5, mine_chr6, mine_chr7, mine_chr8, mine_chr9,
                        mine_chr10, mine_chr11, mine_chr12, mine_chr13, mine_chr14, mine_chr15, mine_chr16, mine_chr17, mine_chr18, mine_chr19, mine_chr20,
                        mine_chr21, mine_chr22, mine_chrX]


# first (authors) dictionary
for n in range(2):
    for i in range(len(cells)):
        for j in range(len(cells)):
            if i==j:
                continue
            chrom = chr_small[n]  # chr14,17
            train_cell = cells[i]
            test_cell = cells[j]

            agg_dir1 = '/scratch/eta/20_june_analysis/authors_dataset'
            pred_dir1 = "/scratch/eta/20_june_analysis/predictions_only_authors"

            agg_train1 = f"{agg_dir1}/{train_cell}_chr{chrom}_feature.csv"
            agg_test1 = f"{agg_dir1}/{test_cell}_chr{chrom}_feature.csv"
            pred_path1 = f"{pred_dir1}/predictions_only_authors_chr{chrom}_train_{train_cell}_test_{test_cell}.csv"

            agg_dir2 = '/scratch/eta/20_june_analysis/largest_authors_dataset_with_dnabert_features'
            pred_dir2 = "/scratch/eta/20_june_analysis/predictions_authors_augmented"

            agg_train2 = f"{agg_dir2}/with_dnabert_features_{train_cell}_chr{chrom}_feature.csv"
            agg_test2 = f"{agg_dir2}/with_dnabert_features_{test_cell}_chr{chrom}_feature.csv"
            pred_path2 = f"{pred_dir2}/predictions_authors_augmented_chr{chrom}_train_{train_cell}_test_{test_cell}.csv"

            #list_of_dictionaries_authors[n]['dnabert'][train_cell][test_cell] = f"{pred_dir2}/predictions_authors_augmented_chr{chrom}_train_{train_cell}_test_{test_cell}.csv"
            list_of_dictionaries_authors[n].setdefault('dnabert', {}).setdefault(train_cell, {}).setdefault(test_cell, {})
            list_of_dictionaries_authors[n]['dnabert'][train_cell][test_cell]= f"{pred_dir2}/predictions_authors_augmented_chr{chrom}_train_{train_cell}_test_{test_cell}.csv"
            #list_of_dictionaries_authors[n]['control'][train_cell][test_cell] = f"{pred_dir1}/predictions_only_authors_chr{chrom}_train_{train_cell}_test_{test_cell}.csv"
            list_of_dictionaries_authors[n].setdefault('control', {}).setdefault(train_cell, {}).setdefault(test_cell, {})
            list_of_dictionaries_authors[n]['control'][train_cell][test_cell] = f"{pred_dir1}/predictions_only_authors_chr{chrom}_train_{train_cell}_test_{test_cell}.csv"

    list_of_dictionaries_authors[n]['save'] = f'/scratch/eta/20_june_analysis/plots/cross_cell_plot_authors_chr{chr_small[n]}'
            

# second (my) dictionary
for n in range(23):
    for i in range(len(cells)):
        for j in range(len(cells)):
            if i==j:
                continue
            chrom = chr_large[n] # chr1~chrX
            train_cell = cells[i]
            test_cell = cells[j]

            agg_dir1 = '/scratch/eta/20_june_analysis/pair_concat_features'
            pred_dir1 = "/scratch/eta/20_june_analysis/predictions_mine_without_dnabert"

            agg_train1 = f"{agg_dir1}/{train_cell}_chr{chrom}_pair_concat.csv"
            agg_test1 = f"{agg_dir1}/{test_cell}_chr{chrom}_pair_concat.csv"
            pred_path1 = f"{pred_dir1}/predictions_mine_without_dnabert_chr{chrom}_train_{train_cell}_test_{test_cell}.csv"
            
            agg_dir2 = '/scratch/eta/20_june_analysis/dnabert_datasets_largest'
            pred_dir2 = "/scratch/eta/20_june_analysis/predictions_mine_with_dnabert"

            agg_train2 = f"{agg_dir2}/with_dnabert_features_{train_cell}_chr{chrom}_pair_concat.csv"
            agg_test2 = f"{agg_dir2}/with_dnabert_features_{test_cell}_chr{chrom}_pair_concat.csv"
            pred_path2 = f"{pred_dir2}/predictions_mine_with_dnabert_chr{chrom}_train_{train_cell}_test_{test_cell}.csv"

            #list_of_dictionaries_mine[n]['dnabert'][train_cell][test_cell] = f"{pred_dir2}/predictions_mine_with_dnabert_chr{chrom}_train_{train_cell}_test_{test_cell}.csv"
            list_of_dictionaries_mine[n].setdefault('dnabert', {}).setdefault(train_cell, {}).setdefault(test_cell, {})
            list_of_dictionaries_mine[n]['dnabert'][train_cell][test_cell] = f"{pred_dir2}/predictions_mine_with_dnabert_chr{chrom}_train_{train_cell}_test_{test_cell}.csv"

            #list_of_dictionaries_mine[n]['control'][train_cell][test_cell] = f"{pred_dir1}/predictions_mine_without_dnabert_chr{chrom}_train_{train_cell}_test_{test_cell}.csv"
            list_of_dictionaries_mine[n].setdefault('control', {}).setdefault(train_cell, {}).setdefault(test_cell, {})
            list_of_dictionaries_mine[n]['control'][train_cell][test_cell] = f"{pred_dir1}/predictions_mine_without_dnabert_chr{chrom}_train_{train_cell}_test_{test_cell}.csv"

    list_of_dictionaries_mine[n]['save'] = f'/scratch/eta/20_june_analysis/plots/cross_cell_plot_mine_chr{chr_large[n]}'


# make list of dictionaries
#list_of_dictionaries_authors = [authors_chr14, authors_chr17]
#list_of_dictionaries_mine = [mine_chr1, mine_chr2, mine_chr3, mine_chr4, mine_chr5, mine_chr6, mine_chr7, mine_chr8, mine_chr9,
#                        mine_chr10, mine_chr11, mine_chr12, mine_chr13, mine_chr14, mine_chr15, mine_chr16, mine_chr17, mine_chr18, mine_chr19, mine_chr20,
#                        mine_chr21, mine_chr22, mine_chrX]

tasks2 = []
for item in list_of_dictionaries_authors:
    tasks2.append((pearson_plots_cross_cell, item))
for item in list_of_dictionaries_mine:
    tasks2.append((pearson_plots_cross_cell, item))
    


def main():
    import os

    # Filter tasks: skip those where the output file already exists
    filtered_tasks = [task for task in tasks if not os.path.exists(task[3])]
    
    # run tasks in parallel - make prediction files
    with Pool(processes=20) as pool:
        pool.map(run_task, filtered_tasks)
        
    #run tasks in parallel - make plots
    with Pool(processes=20) as pool:
        pool.map(run_task, tasks2)


if __name__ == '__main__':
    main()