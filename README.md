# Master_thesis_Eewon_Tai_2025

## About
This is a repository for my code used in writing the thesis 'A machine learning approach for the prediction of genome-wide chromatin interactions'.
The following sections provide a brief description of the code in this repository. Details are provided within the respective Python files.
When I refer to 'the paper' or 'the authors', it means this paper: 
Zhang, S., Chasman, D., Knaack, S. et al. In silico prediction of high-resolution Hi-C interaction matrices. Nat Commun 10, 5449 (2019). https://doi.org/10.1038/s41467-019-13423-8

## validate_paper_results_cross_chr_plots.py
First, the results in the paper were validated using only the dataset provided by the authors.
In this script, the cross-chromosome analysis is conducted.
The Random Forest model was trained on the entire dataset of chromosome 17, and tested on the entire dataset of chromosome 14,
for 5 cell lines. The resulting prediction files were saved, and the distance-stratified pearson's correlation plots were made.
## validate_paper_results_cross_cell_plots.py
In this script, the cross-cell analysis is conducted. The Random Forest model was trained and tested using various combinations of cell types,
but only with chromosome 17. The resulting prediction files were saved, and the distance-stratified pearson's correlation plots were made.

## make_new_datasets_from_scratch.py
Next, new datasets were made using .BigWig signal files and HiC count matrices, since the authors only provided a select number of data.
The PAIR-CONCAT datasets only were made, due to time and resource restrictions.

## generate_dnabert_datasets_faster.py
After that, the datasets made previously were augmented by adding more features derived from DNABERT/PCA.
Because of the large size of the dataset, it was done by making a 'lookup table' and adding the new features to each row of the existing dataset by 'looking it up'.
The resulting datasets have features of the epigenetic markers as well as those derived from DNABERT/PCA.

## Dockerfile, requirements.txt
These files were used to generate a Docker image and container.

## cross_chr_plots_largest_datasets.py
With the newly made datasets, the cross-chromosome plots were generated in order to compare it with the authors' results.

## cross_cell_plots_largest_datasets.py
In the same way, the cross-cell-line plots were generated to compare it with the authors' results.

## no_scaling_RF_training_plots.py
Due to the scaling done in the Random Forest model training process, the dataset might be unnecessarily skewed.
In order to test this, this script was made, which employs no scaling in the model training process.

## train_model_on_entire_new_dataset.py
This is a new approach, to train the model on the datasets of all chromosomes for 4 cell lines, and testing it on each chromosome on the left-out cell line.
The script generates many plots from the analysis.

## scatterplot_analysis_correlation.py
This script makes scatterplots of the true count value versus model predictions, for the authors' data and the new data,
in order to test if the results are significant and not due to outliers, which will negatively impact the calculation of pearson's correlation coefficients.

## null_count.py
This script counts the zero values in the latest datasets, and finds that there are a lot of zeros.

## hic_count_statistics.py
This script calculates statistics of the HiC count columns of the latest datasets, and finds that there are a lot of zeros.

## compare_datasets_authors_mine_statistics.py
This script compares the authors' datasets and the new datasets, to find if there are significant differences that leads to drastic changes in the resulting plots.