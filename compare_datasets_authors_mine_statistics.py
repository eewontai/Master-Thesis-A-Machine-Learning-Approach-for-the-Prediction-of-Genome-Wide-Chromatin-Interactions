#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 13:53:00 2025

@author: eta

Generates a .csv file
Statistics of each column in authors' data and my pair-concat data
To compare the datasets, and see why it is so different
"""

# 1. read authors'data and my data (2 examples) - without DNABERT columns
# 2. make dataframe with statistics + zero count for each feature
# 3. save as .csv file

import pandas as pd

# Load the data
authors_Gm12878_chr14 = pd.read_csv('/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/txt_to_csv_window/Gm12878_chr14_feature.csv')
new_data_Gm12878_chr14 = pd.read_csv('/sybig/home/eta/Masterthesis/scripts/make_new_datasets_from_scratch/pair_concat_features/pair_concat_features/Gm12878_chr14_pair_concat.csv')

# summary statistics for each dataset
authors_stats = authors_Gm12878_chr14.describe().T
new_data_stats = new_data_Gm12878_chr14.describe().T

# Add zero counts
authors_stats['Zero_count_authors'] = (authors_Gm12878_chr14 == 0).sum()
new_data_stats['Zero_count_new'] = (new_data_Gm12878_chr14 == 0).sum()

# Reset index
#authors_stats = authors_stats.reset_index()
#new_data_stats = new_data_stats.reset_index()

# Rename 'index' column to 'Feature' if needed
if 'index' in authors_stats.columns:
    authors_stats = authors_stats.rename(columns={'index': 'Feature'})
if 'index' in new_data_stats.columns:
    new_data_stats = new_data_stats.rename(columns={'index': 'Feature'})

# Rename columns for clarity
authors_stats = authors_stats.rename(columns={
    'mean': 'Mean_authors',
    'min': 'Min_authors',
    '25%': 'Q1_authors',
    '50%': 'Median_authors',
    '75%': 'Q3_authors',
    'max': 'Max_authors',
    'std': 'Std_authors'
})  # zero-count column will still exist without renaming it

new_data_stats = new_data_stats.rename(columns={
    'mean': 'Mean_new',
    'min': 'Min_new',
    '25%': 'Q1_new',
    '50%': 'Median_new',
    '75%': 'Q3_new',
    'max': 'Max_new',
    'std': 'Std_new'
})  # zero-count column will still exist without renaming it

# features of authors dataset
features = ['Ctcf_E', 'Dnase_E', 'H3k27ac_E', 'H3k27me3_E', 'H3k36me3_E', 'H3k4me1_E', 'H3k4me2_E', 'H3k4me3_E', 'H3k79me2_E', 'H3k9ac_E', 'H3k9me3_E', 'H4k20me1_E', 'RAD21_E', 'TBP_E', 
            'Ctcf_P', 'Dnase_P', 'H3k27ac_P', 'H3k27me3_P', 'H3k36me3_P', 'H3k4me1_P', 'H3k4me2_P', 'H3k4me3_P', 'H3k79me2_P', 'H3k9ac_P', 'H3k9me3_P', 'H4k20me1_P', 'RAD21_P', 'TBP_P', 
            'Distance', 'Count']   # 30 items
# features of new dataset
features_new = [
    'H3K27ac_E', 'H3K27me3_E', 'H3K36me3_E', 'H3K4me1_E', 'H3K4me2_E', 'H3K4me3_E', 'H3K79me2_E', 'H3K9ac_E', 'H4K20me1_E', 'H3K9me3_E', 'Ctcf_E', 'DNaseI_E',
    'H3K27ac_P', 'H3K27me3_P', 'H3K36me3_P', 'H3K4me1_P', 'H3K4me2_P', 'H3K4me3_P', 'H3K79me2_P', 'H3K9ac_P', 'H4K20me1_P', 'H3K9me3_P', 'Ctcf_P', 'DNaseI_P',
    'Distance', 'Count']    # 26 items, excluding TBP and RAD21 for 2 regions (-4 items)
feature_dict = {
    'H3K27ac_E': 'H3k27ac_E',
    'H3K27me3_E': 'H3k27me3_E',
    'H3K36me3_E': 'H3k36me3_E',
    'H3K4me1_E': 'H3k4me1_E',
    'H3K4me2_E': 'H3k4me2_E',
    'H3K4me3_E': 'H3k4me3_E',
    'H3K79me2_E': 'H3k79me2_E',
    'H3K9ac_E': 'H3k9ac_E',
    'H4K20me1_E': 'H4k20me1_E',
    'H3K9me3_E': 'H3k9me3_E',
    'Ctcf_E': 'Ctcf_E',
    'DNaseI_E': 'Dnase_E',
    'H3K27ac_P': 'H3k27ac_P',
    'H3K27me3_P': 'H3k27me3_P',
    'H3K36me3_P': 'H3k36me3_P',
    'H3K4me1_P': 'H3k4me1_P',
    'H3K4me2_P': 'H3k4me2_P',
    'H3K4me3_P': 'H3k4me3_P',
    'H3K79me2_P': 'H3k79me2_P',
    'H3K9ac_P': 'H3k9ac_P',
    'H4K20me1_P': 'H4k20me1_P',
    'H3K9me3_P': 'H3k9me3_P',
    'Ctcf_P': 'Ctcf_P',
    'DNaseI_P': 'Dnase_P',
    'Distance': 'Distance',
    'Count': 'Count'
    }
columns = ['Mean_authors', 'Min_authors', 'Q1_authors', 'Median_authors', 'Q3_authors', 'Max_authors', 'Std_authors', 'Zero_count_authors',
           'Mean_new', 'Min_new', 'Q1_new', 'Median_new', 'Q3_new', 'Max_new', 'Std_new', 'Zero_count_new']

df = pd.DataFrame(index=features, columns=columns)
if 'index' in df.columns:
    df = df.rename(columns={'index': 'Feature'})
df.index = features

# delete start1, end1, start2, end2
authors_stats = authors_stats[4:]
new_data_stats = new_data_stats[4:]

authors_stats.index = features
new_data_stats.index = features_new

# Fill it
for r in range(len(features)):
    for c in range(len(columns)):
        feature = features[r]
        col = columns[c]
        
        if c < len(columns) // 2:  # authors
            if feature in authors_stats.index:
                df.loc[feature, col] = authors_stats.loc[feature, col]
        else:  # new data
            if r < len(features_new):
                if features_new[r] in new_data_stats.index:
                    df.loc[feature_dict[features_new[r]], col] = new_data_stats.loc[features_new[r], col]
            else:
                continue


# export as csv file
df.to_csv("/sybig/home/eta/Masterthesis/scripts/dataset_analysis/compare_dataset_authors_mine_statistics.csv", index_label='Feature')

'''
authors features:
start1	end1	start2	end2	Ctcf_E	Dnase_E	H3k27ac_E	H3k27me3_E	H3k36me3_E	H3k4me1_E	H3k4me2_E	H3k4me3_E	H3k79me2_E	H3k9ac_E	H3k9me3_E	H4k20me1_E	RAD21_E	TBP_E	
Ctcf_P	Dnase_P	H3k27ac_P	H3k27me3_P	H3k36me3_P	H3k4me1_P	H3k4me2_P	H3k4me3_P	H3k79me2_P	H3k9ac_P	H3k9me3_P	H4k20me1_P	RAD21_P	TBP_P	Distance	Count
# Dnase_E, Dnase_P
new features:
start1	end1	start2	end2	H3K27ac_E	H3K27me3_E	H3K36me3_E	H3K4me1_E	H3K4me2_E	H3K4me3_E	H3K79me2_E	H3K9ac_E	H4K20me1_E	H3K9me3_E	Ctcf_E	DNaseI_E
	H3K27ac_P	H3K27me3_P	H3K36me3_P	H3K4me1_P	H3K4me2_P	H3K4me3_P	H3K79me2_P	H3K9ac_P	H4K20me1_P	H3K9me3_P	Ctcf_P	DNaseI_P	Distance	Count
# DNaseI_E, DNaseI_P	
'''


# 2. check if authors data have duplicate rows (authors data is larger than mine)
# authors_Gm12878_chr14
# new_data_Gm12878_chr14

authors_data_regions = authors_Gm12878_chr14[:4]
new_data_regions = new_data_Gm12878_chr14[:4]

### for authors data ###
# Count how many times each row appears
duplicate_counts_a = authors_data_regions.value_counts()

# Filter only rows that appear more than once (i.e., are duplicates)
duplicates_with_counts_a = duplicate_counts_a[duplicate_counts_a > 1]

# Convert to DataFrame (optional, for easy viewing)
duplicates_df_a = duplicates_with_counts_a.reset_index()
duplicates_df_a.columns = list(authors_data_regions.columns) + ['Duplicate_Count']

print('Duplicate rows (regions) in authors data: ')
print(duplicates_df_a)


### for new data ###
# Count how many times each row appears
duplicate_counts_n = new_data_regions.value_counts()

# Filter only rows that appear more than once (i.e., are duplicates)
duplicates_with_counts_n = duplicate_counts_n[duplicate_counts_n > 1]

# Convert to DataFrame (optional, for easy viewing)
duplicates_df_n = duplicates_with_counts_n.reset_index()
duplicates_df_n.columns = list(new_data_regions.columns) + ['Duplicate_Count']

print('Duplicate rows (regions) in my data: ')
print(duplicates_df_n)

######### NO DUPLICATE ROWS!

# 3. check if authors data have rows where the regions are flipped

# Assume your dataframe is called df and has columns: start1, end1, start2, end2

# Create a set of tuples in canonical order (sorted regions per row)
canonical_pairs = set()

# Store indices of flipped duplicates
flipped_indices = []

for idx, row in authors_data_regions.iterrows():
    region1 = (row['start1'], row['end1'])
    region2 = (row['start2'], row['end2'])

    # Canonical key for lookup: always store regions in a sorted tuple
    key = (region1, region2)
    flipped_key = (region2, region1)

    if flipped_key in canonical_pairs:
        flipped_indices.append(idx)
    else:
        canonical_pairs.add(key)

# Extract the flipped rows
flipped_rows = authors_data_regions.loc[flipped_indices]
print(flipped_rows)

############ NO FLIPPED ROWS!