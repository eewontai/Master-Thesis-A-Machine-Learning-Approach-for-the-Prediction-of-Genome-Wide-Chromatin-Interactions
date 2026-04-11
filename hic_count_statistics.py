#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 10:29:59 2025

@author: eta
"""
########### make a csv file of hic count value statistics

# collect data from directory and return them
def collect_counts_from_directory(directory, suffix=".csv"):
    import os
    import pandas as pd
    data = []
    labels = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(suffix):
            filepath = os.path.join(directory, filename)
            try:
                df = pd.read_csv(filepath)
                if 'Count' in df.columns:
                    data.append(df['Count'])
                    labels.append(os.path.splitext(filename)[0])  # remove .csv
            except Exception as e:
                print(f"Failed to read {filepath}: {e}")
            print(data)
            print(labels)
    return data, labels

# calculate hic count using data from previous function and output a csv file
def hic_count_csv():
    import pandas as pd
    # --- Input paths ---
    dir1 = "/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest"
    dir2 = "/sybig/home/eta/Masterthesis/scripts/make_new_datasets_from_scratch/pair_concat_features/pair_concat_features"
    output_csv_path = "/sybig/home/eta/Masterthesis/scripts/hic_counts_combined_statistics.csv"
    
    # --- Collect data ---
    data1, labels1 = collect_counts_from_directory(dir1)
    data2, labels2 = collect_counts_from_directory(dir2)
    
    all_data = data1 + data2
    all_labels = labels1 + labels2
    
    all_data = [list(t) for t in all_data]
    
    stats_list = []

    for label, values in zip(all_labels, all_data):
        values = pd.Series(values)

        stats = {
            'Label': label,
            'Count': len(values),
            'Mean': values.mean(),
            'Min': values.min(),
            'Q1': values.quantile(0.25),
            'Median': values.median(),
            'Q3': values.quantile(0.75),
            'Max': values.max(),
            'Std': values.std()
        }

        stats_list.append(stats)

    # Create DataFrame
    df_stats = pd.DataFrame(stats_list)

    # Save to CSV
    df_stats.to_csv(output_csv_path, index=False)
    print(f"Saved summary statistics to: {output_csv_path}")



def main():
    hic_count_csv()

if __name__ == '__main__':
    main()