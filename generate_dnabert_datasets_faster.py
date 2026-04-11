#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 15:12:57 2025

@author: eta

Generate DNABERT datasets with all rows of pair-concat files
Make it faster and more efficient

First, I will make a large dataframe (lookup table) that has columns 'chr', 'start', 'end', 'DNABERT-PCA_1', ..., 'DNABERT-PCA_100',
that contains data for all 23 chromosomes (use same dataframe for all cell lines - one reference genome)

Next, using the lookup table, I will append the DNABERT-PCA columns to pair-concat datasets 
(1. authors' data-chr14/17, 2. my synthetic data-all chr) for each row for all cell lines
"""

# use GPU, DOCKER!!


# pip install torch transformers pysam scikit-learn pandas
##########################################################
import pysam
import torch
import pandas as pd
import numpy as np
import random
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
from multiprocessing import Pool, cpu_count, Semaphore, Process
import pickle
import os
import time
from itertools import product
from torch.cuda.amp import autocast
import multiprocessing as mp


# ---------------------------
# CONFIGURATION
# ---------------------------

FASTA_PATH = '/scratch/eta/ucsc_hg19.fa'
PCA_MODEL_PATH = "/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/pca_model.pkl"
OUTPUT_CSV_PATH = "/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/genome_embeddings.csv"
WINDOW_SIZE = 5000
SAMPLE_FOR_PCA = 1000  # Number of samples to train PCA
output_folder = '/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest'  # for later use
# ---------------------------
# INITIALIZATION
# ---------------------------

print("Loading DNABERT model...")
model_name = '/scratch/eta/dnabert_pytorch_models/6-new-12w-0'  # change path
tokenizer = BertTokenizer.from_pretrained(model_name)  # adjust if different
model = BertModel.from_pretrained(model_name).to('cuda')
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.nn.DataParallel(model)  # wrap for multi-GPU
model = model.to(device)
model.eval()

print("Loading genome file...")
genome_file = pysam.FastaFile(FASTA_PATH)

chromosomes = [str(i) for i in range(1, 23)] + ['X']
cell_lines = ['Gm12878', 'K562', 'Hmec', 'Huvec', 'Nhek']
batch_size = 1024
# ---------------------------
# FUNCTION: Process Region
# ---------------------------
# make k-mers of length 6bp for each input sequence to DNABERT model
def kmerize(sequence, k=6):
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

# sequence is length 500bp, run DNABERT model, return numpy array of 768 CLS token values
def process_sequence(chrom, start, end):
    sequence = genome_file.fetch(f'chr{chrom}', start, start + 500).upper()
    if 'N' in sequence:
        return None
    kmer_seq = kmerize(sequence)
    inputs = tokenizer(kmer_seq, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
    # check for nan values
    if not torch.isnan(cls_embedding).any():
        return cls_embedding.squeeze().cpu().numpy()  # Move back to CPU for further processing
    else:
        return None
    
# ---------------------------
# STEP 1: TRAIN PCA - use one PCA model for all use cases!
# 1) because it reduces running time later
# 2) also because some files have too many zeros, which makes it difficult to always make new pca models
# ---------------------------
# make one model per pair-concat file

def make_pca_model(chrom_num='1'):
    print("Sampling sequences for PCA training...")
    vectors = []
    chrom = chrom_num
    for i in range(10000000):
        start = i * 500
        end = start + 500
        vec = process_sequence(chrom, start, end)
        if vec is not None and len(vec) > 0:
            vectors.append(vec)
        else:
            print(f"No vector returned for {chrom} {start}-{end}")
        if len(vectors) > SAMPLE_FOR_PCA:
            break

    print(f"Collected {len(vectors)} vectors.")

    if len(vectors) == 0:
        raise ValueError("No valid vectors were generated for PCA.")

    vectors = np.array(vectors)

    print(f"Vector array shape: {vectors.shape}")
    print("Any NaNs in vectors?", np.isnan(vectors).any())
    print("Any Infs in vectors?", np.isinf(vectors).any())

    pca_model = PCA(n_components=10)
    pca_model.fit(vectors)

    with open(PCA_MODEL_PATH, "wb") as f:
        pickle.dump(pca_model, f)
    print('PCA model saved.')
    
    return pca_model

# ---------------------------
# STEP 2: EXTRACT ALL REGIONS
# ---------------------------

# get all the 5kb regions in each chromosome
def extract_regions():
    print("Generating region list...")
    region_list = []
    for chrom in chromosomes:
        chrom_length = genome_file.get_reference_length(f'chr{chrom}')
        for start in range(0, chrom_length, WINDOW_SIZE):
            end = start + WINDOW_SIZE
            if end <= chrom_length:
                region_list.append((chrom, start, end))
    
    print(f"Total regions: {len(region_list)}")
    return region_list

# ---------------------------
# STEP 3: PROCESS REGIONS
# ---------------------------

# input: 5kb regions
# for each region, generate DNABERT/PCA derived principal components, stored in a dictionary
# chrom: ex) 'chr1', 'chr2' (not '1', '2')
def process_region(chrom, start, end):
    with open(PCA_MODEL_PATH, "rb") as f:
        pca = pickle.load(f)
    try:
        sequences = []
        # Cut the 5kbp region into 10 segments of 500bp
        for start_input in range(start, end, 500):
            seq = genome_file.fetch(f'chr{chrom}', start_input, start_input + 500).upper()
            if 'N' in seq:
                return None  # skip regions with unknown bases
            sequences.append(kmerize(seq))
        
        # Tokenize batch of sequences with padding (if needed)
        inputs = tokenizer(sequences, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # CLS embeddings for all sequences in batch: shape (batch_size, 768)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # Apply PCA transform to all embeddings at once
        pca_embs = pca.transform(cls_embeddings)  # shape (batch_size, 10)
        
        # Flatten PCA embeddings into columns DNABERT-PCA_0 ... DNABERT-PCA_99 (10 * 10 = 100)
        pca_flat = pca_embs.flatten()
        return {
            "chr": chrom,
            "start": start,
            "end": end,
            **{f"DNABERT-PCA_region{i}_pc{j}": float(pca_flat[i*10+j]) for i in range(10) for j in range(10)}
        }
    except Exception as e:
        print(f"Error processing {chrom}:{start}-{end} -> {e}")
        return None

# get 5kb region list, run process_region function in parallel, add them into a single dataframe (lookup table)
def generate_lookup_table():
    # parallel processing
    NUM_WORKERS = 4  # or use cpu_count() - number of cores/processes to use
    
    #make_pca_model()  # finished
    region_list = extract_regions()
    # Create a list of arguments for starmap
    region_args = [(chrom, start, end) for chrom, start, end in region_list]
    
    # Run in parallel
    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.starmap(process_region, region_args)
    
    # Filter out None results
    results = [r for r in results if r is not None]
    
    # SAVE TO CSV
    results = [r for r in results if r is not None]
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved output to {OUTPUT_CSV_PATH}")
    
    
'''From lookup table, add dnabert-pca features to pair-concat datasets one row at a time'''
# Make 
# 1) authors dataset + dnabert-pca features
# 2) my synthetic pair-concat dataset + dnabert-pca features
    
import pandas as pd
from multiprocessing import Pool, cpu_count

# Globals shared across processes
LOOKUP_DICT = None
EMBEDDING_COLS = None

# Initialize shared data in each process
def init_worker(lookup_dict, embedding_cols):
    global LOOKUP_DICT, EMBEDDING_COLS
    LOOKUP_DICT = lookup_dict
    EMBEDDING_COLS = embedding_cols

# Safely fetch feature values from the lookup
def get_features(key):
    data = LOOKUP_DICT.get(key)
    if data is None:
        return [0] * len(EMBEDDING_COLS)
    return [data[col] for col in EMBEDDING_COLS]

# Process a single row from the pair_concat file
def process_row(row):
    chr_num = row['chr'].replace('chr', '')
    key1 = (chr_num, row['start1'], row['end1'])
    key2 = (chr_num, row['start2'], row['end2'])
    return get_features(key1) + get_features(key2)

# Build lookup dictionary from genome embedding CSV
def setup_lookup(genome_embeddings_path):
    df = pd.read_csv(genome_embeddings_path, dtype={'chr': str})
    df['chr'] = df['chr'].astype(str)
    df = df.set_index(['chr', 'start', 'end'])
    embedding_cols = list(df.columns)
    lookup_dict = df.to_dict(orient='index')
    return lookup_dict, embedding_cols

# Main function to enrich a pair_concat file with DnaBERT features
def add_dnabert_pca_features(pair_concat_path, genome_embeddings_path, save_path):
    df_pair = pd.read_csv(pair_concat_path)
    lookup_dict, embedding_cols = setup_lookup(genome_embeddings_path)

    # Convert each row to a dict for easier access during multiprocessing
    rows = df_pair.to_dict(orient='records')

    num_procs = min(cpu_count() * 2, 32)

    # Multiprocessing pool with shared lookup
    with Pool(processes=num_procs, initializer=init_worker, initargs=(lookup_dict, embedding_cols)) as pool:
        features_list = pool.imap(process_row, rows, chunksize=100)
        features_list = list(features_list)  # Materialize results

    # Create new column names
    col_names = [f'Bin1_{col}' for col in embedding_cols] + [f'Bin2_{col}' for col in embedding_cols]
    df_features = pd.DataFrame(features_list, columns=col_names)

    # Merge features with the original data
    df_final = pd.concat([df_pair.reset_index(drop=True), df_features], axis=1)
    df_final.to_csv(save_path, index=False)




    
def main():
    # first, generate loopup table
    generate_lookup_table()
    
    
    # second, make 
    # 1) authors dataset + dnabert-pca features
    # 2) my synthetic pair-concat dataset + dnabert-pca features
    
    import os
    from pathlib import Path
    
    # Set your fixed paths
    genome_embeddings_path = '/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/genome_embeddings.csv'
    input_dir_authors = '/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/txt_to_csv_window'
    input_dir_synthetic = '/sybig/home/eta/Masterthesis/scripts/make_new_datasets_from_scratch/pair_concat_features/pair_concat_features'
    output_dir_authors = '/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/largest_authors_dataset_with_dnabert_features'
    output_dir_synthetic = '/sybig/home/eta/Masterthesis/scripts/dnabert_datasets/largest'
    
    # test
    #add_dnabert_pca_features('/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/txt_to_csv_window/Gm12878_chr14_feature.csv', genome_embeddings_path, '/sybig/home/eta/Masterthesis/scripts/dataset_analysis/expand_authors_dataset_dnabert/largest_authors_dataset_with_dnabert_features/test1.csv')
    
    # Loop through all CSV files in the directory
    for filename in os.listdir(input_dir_authors):
        if filename.endswith('.csv'):
            pair_concat_path = os.path.join(input_dir_authors, filename)
            output_filename = f'with_dnabert_features_{filename}'
            save_path = os.path.join(output_dir_authors, output_filename)
    
            # Call your function
            add_dnabert_pca_features(pair_concat_path, genome_embeddings_path, save_path)
    
    # Loop through all CSV files in the directory
    for filename in os.listdir(input_dir_synthetic):
        if filename.endswith('.csv'):
            pair_concat_path = os.path.join(input_dir_synthetic, filename)
            output_filename = f'with_dnabert_features_{filename}'
            save_path = os.path.join(output_dir_synthetic, output_filename)
    
            # Call your function
            add_dnabert_pca_features(pair_concat_path, genome_embeddings_path, save_path)
    
    
    
if __name__ == "__main__":
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    mp.set_start_method("spawn", force=True)
    main()