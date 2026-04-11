#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 13:43:52 2025

@author: eta

Make pair-concat datasets, consisting of a region pair, and histone marker/transciption factor datasets for 2 regions, and HiC count (normalized), Distance
"""
# use pybigwig and hicstraw to generate new training and test datasets from scratch, replicating the methods used in the paper

###### pip install hic-straw
###### pip install pyBigWig pandas numpy

import hicstraw
import pyBigWig

import numpy as np
import pandas as pd
import concurrent.futures
from functools import partial

import os
from hicstraw import straw
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import csv
from joblib import Parallel, delayed


###################################################################
# make aggregated features of ChIP-seq for 5 cell lines and 23 chromosomes individually
# outputs 5*23 csv files
###################################################################

# Path to the bigWig files for different cell lines and ChIP-seq marks
cell_lines = ['Gm12878', 'K562', 'Hmec', 'Huvec', 'Nhek']
#cell_lines = ['Nhek']
#cell_lines = ['Gm12878']

marks = ['H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K79me2', 'H3K9ac', 'H4K20me1', 'H3K9me3', 'Ctcf', 'DNaseI']
chromosomes = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chrX', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 
               'chr15', 'chr16', 'chr17', 'chr18', 'chr20', 'chr19', 'chr22', 'chr21']
chromosomes_num = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']

# Mapping of cell line and marks to their bigWig file paths
bigwig_paths_Gm12878 = {
    'Gm12878_Ctcf': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneGm12878CtcfStdSig.bigWig',
    'Gm12878_H3K27ac': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneGm12878H3k27acStdSig.bigWig',
    'Gm12878_H3K27me3': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneGm12878H3k27me3StdSig.bigWig',
    'Gm12878_H3K36me3': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneGm12878H3k36me3StdSig.bigWig',
    'Gm12878_H3K4me1': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneGm12878H3k4me1StdSig.bigWig',
    'Gm12878_H3K4me2': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneGm12878H3k4me2StdSig.bigWig',
    'Gm12878_H3K4me3': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneGm12878H3k4me3StdSig.bigWig',
    'Gm12878_H3K79me2': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneGm12878H3k79me2StdSig.bigWig',
    'Gm12878_H3K9ac': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneGm12878H3k9acStdSig.bigWig',
    'Gm12878_H3K9me3': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneGm12878H3k9me3StdSig.bigWig',
    'Gm12878_H4K20me1': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneGm12878H4k20me1StdSig.bigWig',
    'Gm12878_DNaseI': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeOpenChromDnaseGm12878Sig.bigWig'
    }
bigwig_paths_Hmec = {
    'Hmec_Ctcf': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneHmecCtcfStdSig.bigWig',
    'Hmec_H3K9me3': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneHmecH3k09me3Sig.bigWig',
    'Hmec_H3K27ac': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneHmecH3k27acStdSig.bigWig',
    'Hmec_H3K27me3': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneHmecH3k27me3StdSig.bigWig',
    'Hmec_H3K36me3': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneHmecH3k36me3StdSig.bigWig',
    'Hmec_H3K4me1': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneHmecH3k4me1StdSig.bigWig',
    'Hmec_H3K4me2': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneHmecH3k4me2StdSig.bigWig',
    'Hmec_H3K4me3': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneHmecH3k4me3StdSig.bigWig',
    'Hmec_H3K79me2': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneHmecH3k79me2Sig.bigWig',
    'Hmec_H3K9ac': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneHmecH3k9acStdSig.bigWig',
    'Hmec_H4K20me1': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneHmecH4k20me1StdSig.bigWig',
    'Hmec_DNaseI': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeOpenChromDnaseHmecSig.bigWig'
    }
bigwig_paths_Huvec = {
    'Huvec_Ctcf': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneHuvecCtcfStdSig.bigWig',
    'Huvec_H3K9me3': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneHuvecH3k09me3Sig.bigWig',
    'Huvec_H3K27ac': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneHuvecH3k27acStdSig.bigWig',
    'Huvec_H3K27me3': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneHuvecH3k27me3StdSig.bigWig',
    'Huvec_H3K36me3': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneHuvecH3k36me3StdSig.bigWig',
    'Huvec_H3K4me1': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneHuvecH3k4me1StdSig.bigWig',
    'Huvec_H3K4me2': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneHuvecH3k4me2StdSig.bigWig',
    'Huvec_H3K4me3': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneHuvecH3k4me3StdSig.bigWig',
    'Huvec_H3K79me2': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneHuvecH3k79me2Sig.bigWig',
    'Huvec_H3K9ac': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneHuvecH3k9acStdSig.bigWig',
    'Huvec_H4K20me1': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneHuvecH4k20me1StdSig.bigWig',
    'Huvec_DNaseI': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeOpenChromDnaseHuvecSig.bigWig'
    }
bigwig_paths_K562 = {
    'K562_Ctcf': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneK562CtcfStdSig.bigWig',
    'K562_H3K27ac': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneK562H3k27acStdSig.bigWig',
    'K562_H3K27me3': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneK562H3k27me3StdSig.bigWig',
    'K562_H3K36me3': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneK562H3k36me3StdSig.bigWig',
    'K562_H3K4me1': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneK562H3k4me1StdSig.bigWig',
    'K562_H3K4me2': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneK562H3k4me2StdSig.bigWig',
    'K562_H3K4me3': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneK562H3k4me3StdSig.bigWig',
    'K562_H3K79me2': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneK562H3k79me2StdSig.bigWig',
    'K562_H3K9ac': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneK562H3k9acStdSig.bigWig',
    'K562_H3K9me3': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneK562H3k9me3StdSig.bigWig',
    'K562_H4K20me1': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneK562H4k20me1StdSig.bigWig',
    'K562_DNaseI': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeOpenChromDnaseK562Sig.bigWig'
    }
    
bigwig_paths_Nhek = {
    'Nhek_Ctcf': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneNhekCtcfStdSig.bigWig',
    'Nhek_H3K9me3': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneNhekH3k09me3Sig.bigWig',
    'Nhek_H3K27ac': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneNhekH3k27acStdSig.bigWig',
    'Nhek_H3K27me3': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneNhekH3k27me3StdSig.bigWig',
    'Nhek_H3K36me3': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneNhekH3k36me3StdSig.bigWig',
    'Nhek_H3K4me1': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneNhekH3k4me1StdSig.bigWig',
    'Nhek_H3K4me2': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneNhekH3k4me2StdSig.bigWig',
    'Nhek_H3K4me3': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneNhekH3k4me3StdSig.bigWig',
    'Nhek_H3K79me2': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneNhekH3k79me2Sig.bigWig',
    'Nhek_H3K9ac': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneNhekH3k9acStdSig.bigWig',
    'Nhek_H4K20me1': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeBroadHistoneNhekH4k20me1StdSig.bigWig',
    'Nhek_DNaseI': '/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/wgEncodeOpenChromDnaseNhekSig.bigWig'
    }

# check which chromosomes all the bigwig files have in common
def common_chromosomes():
    import pyBigWig
    import glob
    
    # Path to your BigWig files — adjust the pattern as needed
    bigwig_files = glob.glob("/scratch/eta/make_new_datasets_from_scratch/bigwig_data_download/*.bigWig")
    
    # Set to keep common chromosomes
    common_chroms = None
    
    for bw_file in bigwig_files:
        with pyBigWig.open(bw_file) as bw:
            chroms_in_file = set(bw.chroms().keys())
            if common_chroms is None:
                common_chroms = chroms_in_file
            else:
                common_chroms &= chroms_in_file  # intersect
    
    print("Common chromosomes across all BigWig files:")
    print(sorted(common_chroms))
    # ['chr1', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr2', 'chr20', 'chr21', 'chr22', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chrX']



# Function to read chromosome sizes from a file
def read_chromosome_sizes(file_path):
    chrom_sizes = {}
    with open(file_path, 'r') as f:
        for line in f:
            chrom, size = line.strip().split()
            chrom_sizes[chrom] = int(size)
    return chrom_sizes

# Function to generate 5000bp bins for each chromosome
def generate_bins(chrom_sizes, bin_size=5000):
    bins = []
    for chrom, size in chrom_sizes.items():
        for start in range(0, size, bin_size):
            end = min(start + bin_size, size)
            bins.append((chrom, start, end))
    print("First few bins:", bins[:5])
    return bins

# Function to get the signal from a bigWig file for each bin
# sum all the signals into one bin
def get_bin_signals(bw_file, bins, chromosome):
    """
    Extracts signal values from a BigWig file for the specified chromosome only.
    
    Parameters:
    - bw_file: path to the BigWig file
    - bins: list of (chrom, start, end) tuples
    - chromosome: target chromosome string, e.g. "chr1"
    
    Returns:
    - signals: list of signal values for bins on the given chromosome
    """
    chrom_sizes_file = '/scratch/eta/make_new_datasets_from_scratch/hg19.chrom.sizes'
    chrom_sizes = read_chromosome_sizes(chrom_sizes_file)
    bw = pyBigWig.open(bw_file)
    print("BigWig chroms:", bw.chroms().keys())  # Debugging

    signals = []

    # Filter bins for the target chromosome
    chrom_bins = [b for b in bins if b[0] == chromosome]

    for chrom, start, end in chrom_bins:
        if start < 0 or end <= start:
            print(f"Skipping invalid bin: {chrom}:{start}-{end}")
            continue
        if chrom not in bw.chroms():
            print(f"Skipping missing chrom in bigWig: {chrom}")
            continue
        if end > chrom_sizes.get(chrom, 0):
            print(f"Skipping out-of-bound bin: {chrom}:{start}-{end}")
            continue
        try:
            values = bw.values(chrom, start, end, numpy=True)
            sum_signal = np.nansum(values)
            signals.append(sum_signal)
        except RuntimeError as e:
            print(f"Error reading values for {chrom}:{start}-{end} - {str(e)}")
            signals.append(np.nan)  # or continue
    bw.close()
    return signals



# Normalize signal by sequencing depth (Counts per Million, CPM)
def normalize_signal(signals):
    total_signal = np.sum(signals)
    return [x / total_signal * 1e6 for x in signals]

# for each chromosome and cell line, return a row that contains chromosome, start and end position for one region, and normalized bigwig signals for the epigenetic markers
def process_chromosome_cell_line(cell_line, chromosome, bigwig_paths, marks, bins):
    # Filter bins for this chromosome
    chr_bins = [b for b in bins if b[0] == chromosome]

    # Start with chr/start/end
    df = pd.DataFrame({
        'chr': [b[0] for b in chr_bins],
        'start': [b[1] for b in chr_bins],
        'end': [b[2] for b in chr_bins]
    })

    for mark in marks:
        bw_file = bigwig_paths[f"{cell_line}_{mark}"]  # individual bigwig file

        # Get signals and normalize
        signals = get_bin_signals(bw_file, chr_bins, chromosome)
        normalized_signals = normalize_signal(signals)

        # Add column for this mark
        df[mark] = normalized_signals

    return (cell_line, chromosome, df)


# make csv files for each chromosome and cell line that contains chromosome, start and end position for one region, and normalized bigwig signals for the epigenetic markers for each 5kb region
def feature_extraction_and_representation():
    # pip install pyBigWig pandas numpy
    # Path to the chromosome sizes file
    chrom_sizes_file = '/scratch/eta/make_new_datasets_from_scratch/hg19.chrom.sizes'
    
    # Read chromosome sizes from the file
    chrom_sizes = read_chromosome_sizes(chrom_sizes_file)
    
    # Generate 5000bp bins for each chromosome
    bins = generate_bins(chrom_sizes, bin_size=5000)

    dfs = {} # empty dictionary to collect everything
    
    # Now loop and update using parallelization
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        # Submit tasks to the executor
        futures = []
        for cell_line in cell_lines:
            bigwig_paths = globals().get(f"bigwig_paths_{cell_line}")
            for chromosome in chromosomes:
                # Submit the task for each combination of cell line and chromosome
                futures.append(executor.submit(process_chromosome_cell_line, cell_line, chromosome, bigwig_paths, marks, bins))
    
        for future in concurrent.futures.as_completed(futures):
            cell_line, chromosome, df = future.result()
            if cell_line not in dfs:
                dfs[cell_line] = {}
            dfs[cell_line][chromosome] = df  # full DataFrame with signals is here
    
    
    # Collapse replicates by row-wise median and keep only one column per mark
    for cell_line in cell_lines:
        for chromosome in chromosomes:
            df = dfs[cell_line][chromosome]
    
            for mark in marks:
                # Get all replicate columns: 'H3K27ac', 'H3K27ac_rep1', etc.
                replicate_cols = [col for col in df.columns if col == mark or col.startswith(f"{mark}_rep")]
    
                if len(replicate_cols) > 1:
                    # Compute median across replicate columns
                    df[f"{mark}_median"] = df[replicate_cols].median(axis=1)
    
                    # Drop the original replicate columns
                    df.drop(columns=replicate_cols, inplace=True)

    
    # Save one CSV per cell line and chromosome
    output_dir = '/scratch/eta/make_new_datasets_from_scratch/aggregated_features_5kb/'
    
    for cell_line in cell_lines:
        for chromosome in chromosomes:
            df = dfs[cell_line][chromosome]
            output_path = f"{output_dir}{cell_line}_{chromosome}_aggregated_chipseq_features.csv"
            df.to_csv(output_path, index=False)

##########################################
# generate pair-concat features
# use same method as the paper when generating hic values
########################################################
MAX_DISTANCE = 1_000_000  # maximum genomic distance between two regions

cell_lines_hic_folder = ['GM12878_combined', 'HMEC', 'HUVEC', 'K562', 'NHEK']
#cell_lines_hic_folder = ['HMEC', 'HUVEC', 'NHEK']

# generate all hic dataframe files in one location (RAWobserved divided by SQRTVCnorm)
# parallel processing
def process_one_hic(cell_line, chromosome, hic_path):
    try:
        raw_observed = pd.read_csv(
            f'/scratch/eta/make_new_datasets_from_scratch/hic_data_download/hic_data_download/{cell_line}/5kb_resolution_intrachromosomal/{chromosome}/MAPQGE30/{chromosome}_5kb.RAWobserved',
            sep='\s+', header=None
        )

        sqrtvc_norm = pd.read_csv(
            f'/scratch/eta/make_new_datasets_from_scratch/hic_data_download/hic_data_download/{cell_line}/5kb_resolution_intrachromosomal/{chromosome}/MAPQGE30/{chromosome}_5kb.SQRTVCnorm',
            header=None
        )[0]  # first column as a Series

        # Calculate bin indices for start1 and start2
        bin1 = (raw_observed.iloc[:, 0] // 5000).astype(int)
        bin2 = (raw_observed.iloc[:, 1] // 5000).astype(int)

        # Get corresponding normalization values
        norm1 = sqrtvc_norm.iloc[bin1.values].values
        norm2 = sqrtvc_norm.iloc[bin2.values].values

        # Calculate normalization factors
        normalization_factors = norm1 * norm2

        # Apply normalization, avoiding division by zero
        mask = (norm1 != 0) & (norm2 != 0)
        raw_observed.loc[mask, 2] = raw_observed.loc[mask, 2] / normalization_factors[mask]
        raw_observed.loc[~mask, 2] = 0

        # Make sure output folder exists
        os.makedirs(hic_path, exist_ok=True)

        # Save
        raw_observed.to_csv(f'{hic_path}/hic_calculated_{cell_line}_{chromosome}.csv', index=False, header=False)

    except FileNotFoundError:
        print(f"Missing file for {cell_line} {chromosome}, skipping.")


def hic_calculate(hic_path):
    Parallel(n_jobs=23)(  # or however many cores you want
        delayed(process_one_hic)(cell_line, chromosome, hic_path)
        for cell_line in cell_lines_hic_folder
        for chromosome in chromosomes
    )

# using the region-level feature aggregtion data (generated above), and hic count values (generated above),
# make pair-concat datasets, containing a region pair, and histone marker/transciption factor datasets for 2 regions, and HiC count (normalized), Distance
def make_pair_concat_datasets_with_new_data(cell_line, chromosome, chipseq_path, hic_path, output_path):
    chipseq_df = pd.read_csv(chipseq_path)
    
    if cell_line == 'Gm12878':
        hic_df = pd.read_csv(f'{hic_path}/hic_calculated_GM12878_combined_{chromosome}.csv')
    if cell_line == 'K562':
        hic_df = pd.read_csv(f'{hic_path}/hic_calculated_K562_{chromosome}.csv')
    if cell_line == 'Nhek':
        hic_df = pd.read_csv(f'{hic_path}/hic_calculated_NHEK_{chromosome}.csv')
    if cell_line == 'Huvec':
        hic_df = pd.read_csv(f'{hic_path}/hic_calculated_HUVEC_{chromosome}.csv')
    if cell_line == 'Hmec':
        hic_df = pd.read_csv(f'{hic_path}/hic_calculated_HMEC_{chromosome}.csv')
        
    hic_df.set_index([hic_df.columns[0], hic_df.columns[1]], inplace=True)
    
    # Drop unnecessary columns to reduce memory
    regions = chipseq_df.drop(columns=['chr'])
    histone_cols = [col for col in regions.columns if col not in ['start', 'end']]
    num_regions = len(regions)

    # Convert to numpy arrays for efficient access
    starts = regions['start'].values
    ends = regions['end'].values
    features = regions[histone_cols].values
    
    # Write output with low-overhead csv writer
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = (
            ['chr', 'start1', 'end1', 'start2', 'end2'] +
            [f"{col}_E" for col in histone_cols] +
            [f"{col}_P" for col in histone_cols] +
            ['Distance', 'Count']
        )
        writer.writerow(header)
        for i in range(num_regions):
            start1, end1 = starts[i], ends[i]
            row1_features = features[i]

            # Convert genomic start to bin index
            bin_index1 = start1

            for j in range(i + 1, num_regions):
                start2, end2 = starts[j], ends[j]
                if start2 - end1 > MAX_DISTANCE:
                    break

                row2_features = features[j]

                # Convert genomic start to bin index
                bin_index2 = start2
                
                #### find value in hic_df ###
                try:
                    count = hic_df.loc[(start1, start2)].iloc[0]
                except KeyError:
                    count = 0


                row = [
                    chromosome, start1, end1, start2, end2,
                    *row1_features, *row2_features,
                    start2 - end1, count
                ]
                writer.writerow(row)

    print(f"[✓] Finished: {output_path}")


# parallel processing
def make_pair_concat_new_parallel(cell_lines, chromosomes, chipseq_data_dir, hic_data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    futures = []
    max_workers = 23  # adjust based on memory per core

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for cell_line in cell_lines:
            for chromosome in chromosomes:
                chipseq_file = os.path.join(chipseq_data_dir, f"{cell_line}_{chromosome}_aggregated_chipseq_features.csv")
                hic_path = '/scratch/eta/make_new_datasets_from_scratch/hic_data_download/'
                output_file = os.path.join(output_dir, f"{cell_line}_{chromosome}_pair_concat.csv")

                if not os.path.exists(chipseq_file):
                    print(f"Missing: {chipseq_file}, skipping.")
                    continue
                try:
                    # Don't pass large DataFrames, pass only paths
                    futures.append(executor.submit(
                        make_pair_concat_datasets_with_new_data,
                        cell_line,
                        chromosome,
                        chipseq_file,
                        hic_path,
                        output_file
                    ))
                except Exception as e:
                    print(f"Error processing task: {e}")
                    raise  # Re-raise the exception so the pool can properly handle it
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"[✗] Worker crashed: {type(e).__name__}: {e}")

'''
cell_lines = ['Gm12878', 'K562', 'Hmec', 'Huvec', 'Nhek']
chromosomes = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chrX', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 
               'chr15', 'chr16', 'chr17', 'chr18', 'chr20', 'chr19', 'chr22', 'chr21']
chromosomes_num = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']
'''
    

def main():
    common_chromosomes()  # check which chromosomes all the bigwig files have in common
    feature_extraction_and_representation()  # make datasets of regions and histone marker/TF values
    
    chipseq_data_dir = '/scratch/eta/make_new_datasets_from_scratch/aggregated_features_5kb/aggregated_features_5kb'
    hic_data_dir = '/scratch/eta/make_new_datasets_from_scratch/hic_data_download/'
    output_dir = '/scratch/eta/make_new_datasets_from_scratch/pair_concat_features/'
    
    # make pair concat datasets with new data from authors reply
    hic_calculate(hic_data_dir)
    make_pair_concat_new_parallel(cell_lines, chromosomes, chipseq_data_dir, hic_data_dir, output_dir)
    # has a problem here (probably in parallel processing) - had to generate dataset individually for all cell lines
    
    return 0

if __name__ == "__main__":
    main()
