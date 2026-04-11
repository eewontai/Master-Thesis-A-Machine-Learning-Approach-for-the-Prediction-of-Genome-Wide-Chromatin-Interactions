#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 11:35:45 2025

@author: eta
"""

import multiprocessing
import os
    

# change predictions file (results from the paper) format, text to csv
def predictions_txt_to_csv(text_file_path, save_path_csv):
    import pandas as pd
    df = pd.read_csv(text_file_path, sep='\s+')
    # Use regex to extract new columns
    new_cols = df['Column'].str.extract(r'(chr\d+)_(\d+)_(\d+)-chr\d+_(\d+)_(\d+)', expand=True)
    new_cols.columns = ['chr', 'start1', 'end1', 'start2', 'end2']
    # Drop the 'Pair' column
    df = df.drop(columns=['Column'])
    # Concatenate new columns at the front
    df = pd.concat([new_cols, df], axis=1)
    
    df.to_csv(save_path_csv, index=False)



def run_task(args):
    input_path, output_path = args
    predictions_txt_to_csv(input_path, output_path)
    

# for each prediction file, make 11 scatterplots (Distance = 0, 0.1, ..., 0.9, 1)
# csv_path is path to predictions csv file
# output folder is where I will store the output png files
def scatterplot_correlation(csv_path, output_folder, cell_line):
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    df = pd.read_csv(csv_path)

    distance_values = [i * 100000 for i in range(11)]  # 0 to 1,000,000 in steps of 100,000

    for distance in distance_values:
        # Filter rows for current distance (allowing small tolerance for float comparison)
        filtered_df = df[abs(df['Distance'] - distance) < 1e-6]

        # Skip if there's no data for this distance
        if filtered_df.empty:
            print(f"No data found for Distance = {distance}")
            continue

        # Plot
        plt.figure(figsize=(7, 7))
        plt.scatter(
            filtered_df['TrueValue'],
            filtered_df['PredictedValue'],
            alpha=0.6,
            color="#87CEEB",  # Sky blue
            edgecolors="k",   # Optional: add black edge for contrast
            linewidths=0.2
        )
        plt.title(f"True vs Predicted (Distance = {distance}, {cell_line})", fontsize=14)
        plt.xlabel("TrueValue", fontsize=12)
        plt.ylabel("PredictedValue", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True)
        plt.tight_layout()

        # Save plot
        filename = f"{cell_line}_chr14_scatter_distance_{distance/1000000}_Mbp.png"
        filepath = os.path.join(output_folder, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Saved: {filepath}")

    
    
def main():
    base_save_path = '/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/txt_to_csv_predictions'

    tasks = [
        ('/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/WINDOW/chr14/CV/Gm12878_chr14_RF_tree20_prediction.txt',
         os.path.join(base_save_path, 'Gm12878_chr14_predictions.csv')),

        ('/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/WINDOW/chr17/CV/Gm12878_chr17_RF_tree20_prediction.txt',
         os.path.join(base_save_path, 'Gm12878_chr17_predictions.csv')),

        ('/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/WINDOW/chr14/CV/Hmec_chr14_RF_tree20_prediction.txt',
         os.path.join(base_save_path, 'Hmec_chr14_predictions.csv')),

        ('/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/WINDOW/chr17/CV/Hmec_chr17_RF_tree20_prediction.txt',
         os.path.join(base_save_path, 'Hmec_chr17_predictions.csv')),

        ('/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/WINDOW/chr14/CV/Huvec_chr14_RF_tree20_prediction.txt',
         os.path.join(base_save_path, 'Huvec_chr14_predictions.csv')),

        ('/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/WINDOW/chr17/CV/Huvec_chr17_RF_tree20_prediction.txt',
         os.path.join(base_save_path, 'Huvec_chr17_predictions.csv')),

        ('/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/WINDOW/chr14/CV/K562_chr14_RF_tree20_prediction.txt',
         os.path.join(base_save_path, 'K562_chr14_predictions.csv')),

        ('/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/WINDOW/chr17/CV/K562_chr17_RF_tree20_prediction.txt',
         os.path.join(base_save_path, 'K562_chr17_predictions.csv')),

        ('/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/WINDOW/chr14/CV/Nhek_chr14_RF_tree20_prediction.txt',
         os.path.join(base_save_path, 'Nhek_chr14_predictions.csv')),

        ('/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/WINDOW/chr17/CV/Nhek_chr17_RF_tree20_prediction.txt',
         os.path.join(base_save_path, 'Nhek_chr17_predictions.csv'))
    ]

    # Limit number of worker processes if needed
    num_workers = 10
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(run_task, tasks)
    
    scatterplot_correlation('/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/txt_to_csv_predictions/Gm12878_chr14_predictions.csv', '/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/scatterplots_truevalue_predictedvalue', 'Gm12878')
    scatterplot_correlation('/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/txt_to_csv_predictions/K562_chr14_predictions.csv', '/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/scatterplots_truevalue_predictedvalue', 'K562')        
    scatterplot_correlation('/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/txt_to_csv_predictions/Hmec_chr14_predictions.csv', '/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/scatterplots_truevalue_predictedvalue', 'Hmec')        
    scatterplot_correlation('/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/txt_to_csv_predictions/Huvec_chr14_predictions.csv', '/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/scatterplots_truevalue_predictedvalue', 'Huvec')        
    scatterplot_correlation('/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/txt_to_csv_predictions/Nhek_chr14_predictions.csv', '/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/scatterplots_truevalue_predictedvalue', 'Nhek')        
    scatterplot_correlation('/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest/predictions_plots/predictions_mine_pc_Gm12878_dnabert_train17_test14.csv', '/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest/predictions_plots', 'Gm12878')
    scatterplot_correlation('/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest/predictions_plots/predictions_mine_pc_K562_dnabert_train17_test14.csv', '/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest/predictions_plots', 'K562')        
    scatterplot_correlation('/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest/predictions_plots/predictions_mine_pc_Hmec_dnabert_train17_test14.csv', '/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest/predictions_plots', 'Hmec')        
    scatterplot_correlation('/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest/predictions_plots/predictions_mine_pc_Huvec_dnabert_train17_test14.csv', '/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest/predictions_plots', 'Huvec')        
    scatterplot_correlation('/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest/predictions_plots/predictions_mine_pc_Nhek_dnabert_train17_test14.csv', '/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest/predictions_plots', 'Nhek')        

if __name__ == '__main__':
    main()