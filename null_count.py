#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 14:27:04 2025

@author: eta

Count zero values in new datasets with added dnabert columns
"""
import csv
import os
from collections import Counter

# Directory with CSV files
csv_dir = '/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest'
csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

# Output path for summary CSV
summary_output = '/sybig/home/eta/Masterthesis/scripts/zero_counts_summary.csv'

# Store summary for each file
summary_data = []

for filename in csv_files:
    path = os.path.join(csv_dir, filename)
    zero_counts = Counter()

    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header

        for row in reader:
            try:
                numeric_row = [float(val) for val in row[1:] if val.strip() != '']
            except ValueError:
                continue  # Skip rows with invalid data

            num_zeros = numeric_row.count(0.0)
            if num_zeros <= 10:
                zero_counts[num_zeros] += 1
            else:
                zero_counts['>10'] += 1

    # Format output row: filename + 0-10 + >10
    row_data = [filename] + [zero_counts[i] for i in range(11)] + [zero_counts['>10']]
    summary_data.append(row_data)

# Write the summary file
with open(summary_output, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['Filename'] + [f'Zeros_{i}' for i in range(11)] + ['Zeros_>10']
    writer.writerow(header)
    writer.writerows(summary_data)

print(f"[✓] Summary saved to: {summary_output}")
