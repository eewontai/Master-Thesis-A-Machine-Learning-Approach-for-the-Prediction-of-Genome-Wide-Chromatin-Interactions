# validate the results of the paper, cross-cell analysis and plotting
# train/test on different cell lines, same chromosome (chr17)
# generate 1 plot, made of 5*5 subplots

# code details are mostly similar with validate_paper_results_cross_chr_plots.py

# generate many different prediction .csv files in parallel

# get text files, train/test model, save prediction file in path
# path: complete path to the file that will be saved as predictions, including filename.csv, in string form
def crossCell_pair_concat(file_train, file_test, path):
    # import libraries
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    df_train = pd.read_csv(file_train, sep='\s+').drop(['Ctcf_W', 'Dnase_W', 'H3k27ac_W', 'H3k27me3_W', 'H3k36me3_W', 'H3k4me1_W', 'H3k4me2_W', 'H3k4me3_W', 'H3k79me2_W', 'H3k9ac_W', 'H3k9me3_W', 'H4k20me1_W', 'RAD21_W', 'TBP_W'], axis=1)
    df_test = pd.read_csv(file_test, sep='\s+').drop(['Ctcf_W', 'Dnase_W', 'H3k27ac_W', 'H3k27me3_W', 'H3k36me3_W', 'H3k4me1_W', 'H3k4me2_W', 'H3k4me3_W', 'H3k79me2_W', 'H3k9ac_W', 'H3k9me3_W', 'H4k20me1_W', 'RAD21_W', 'TBP_W'], axis=1)

    # model training - scikit learn random forest regression
    X_train = df_train.iloc[:, 1:(len(df_train.columns)-1)]
    y_train = df_train[['Count']]
    X_test = df_test.iloc[:, 1:(len(df_test.columns)-1)]
    y_test = df_test[['Count']]

    # scale and normalize columns in X, since Distance column has very high values compared to other columns
    # Initialize the scaler
    scaler = StandardScaler()
    scaler.fit(X_train)  # Fit on train set only

    # Transform train and test sets
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Do not fit again!
    
    # Transform y from column vector to 1D array
    y_train_array = y_train.to_numpy().flatten()
    y_test_array = y_test.to_numpy().flatten()

    # Define the model
    # n_estimators: default=100. the number of trees in the forest
    model = RandomForestRegressor(n_estimators=20, max_features=0.3, bootstrap=True, random_state=42)

    model.fit(X_train_scaled, y_train_array)
    predictions = pd.DataFrame({
        'Pair': df_test['Pair'],
        'TrueValue': y_test_array,
        'PredictedValue': model.predict(X_test_scaled),
        'Distance': df_test['Distance']
        })

    # save predictions to a file
    predictions.to_csv(path, index=False)


# get text files, train/test model, save prediction file in path
# path: complete path to the file that will be saved as predictions, including filename.csv, in string form
def crossCell_window(file_train, file_test, path):
    # import libraries
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    
    df_train = pd.read_csv(file_train, sep='\s+')
    df_test = pd.read_csv(file_test, sep='\s+')
    
    # model training - scikit learn random forest regression
    X_train = df_train.iloc[:, 1:(len(df_train.columns)-1)]
    y_train = df_train[['Count']]
    X_test = df_test.iloc[:, 1:(len(df_test.columns)-1)]
    y_test = df_test[['Count']]
    
    # scale and normalize columns in X, since Distance column has very high values compared to other columns
    # Initialize the scaler
    scaler = StandardScaler()
    scaler.fit(X_train)  # Fit on train set only

    # Transform train and test sets
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Do not fit again!

    # Transform y from column vector to 1D array
    y_train_array = y_train.to_numpy().flatten()
    y_test_array = y_test.to_numpy().flatten()

    # Define the model
    # n_estimators: default=100. the number of trees in the forest
    model = RandomForestRegressor(n_estimators=20, max_features=0.3, bootstrap=True, random_state=42)
    

    model.fit(X_train_scaled, y_train_array)
    predictions = pd.DataFrame({
        'Pair': df_test['Pair'],
        'TrueValue': y_test_array,
        'PredictedValue': model.predict(X_test_scaled),
        'Distance': df_test['Distance']
        })

    # save predictions to a file (predictions_2nd)
    predictions.to_csv(path, index=False)


# get text files, train/test model, save prediction file in path
# path: complete path to the file that will be saved as predictions, including filename.csv, in string form
def crossCell_multicell(file_train, file_test, path):
    # import libraries
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    
    df_train = pd.read_csv(file_train, sep='\s+')
    df_test = pd.read_csv(file_test, sep='\s+')

    # model training - scikit learn random forest regression
    X_train = df_train.iloc[:, 1:(len(df_train.columns)-1)]
    y_train = df_train[['Count']]
    X_test = df_test.iloc[:, 1:(len(df_test.columns)-1)]
    y_test = df_test[['Count']]

    # scale and normalize columns in X, since Distance column has very high values compared to other columns
    # Initialize the scaler
    scaler = StandardScaler()
    scaler.fit(X_train)  # Fit on train set only

    # Transform train and test sets
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Do not fit again!

    
    # Transform y from column vector to 1D array
    y_train_array = y_train.to_numpy().flatten()
    y_test_array = y_test.to_numpy().flatten()

    # Define the model
    # n_estimators: default=100. the number of trees in the forest
    model = RandomForestRegressor(n_estimators=20, max_features=0.3, bootstrap=True, random_state=42)
    

    model.fit(X_train_scaled, y_train_array)
    predictions = pd.DataFrame({
        'Pair': df_test['Pair'],
        'TrueValue': y_test_array,
        'PredictedValue': model.predict(X_test_scaled),
        'Distance': df_test['Distance']
        })

    # save predictions to a file (predictions_2nd)
    predictions.to_csv(path, index=False)




# generate 1 plot, made of 5 subplots
def pearson_plots():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt#
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
            pair_concat = pd.read_csv("/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_" + order[column] + "_" + order[row] + "_pc_chr17.csv")
            window = pd.read_csv("/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_" + order[column] + "_" + order[row] + "_w_chr17.csv")
            multicell = pd.read_csv("/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_" + order[column] + "_" + order[row] + "_mc_chr17.csv")

            pearson_array_pair_concat = np.zeros(200)
            pearson_array_window = np.zeros(200)
            pearson_array_multicell = np.zeros(200)
            
            # >0, <=5000
            # >5000, <=10,000
            # ...
            # >995,000 , <=1,000,000
            # total 200 bins
            for i in range(200):
                min_value = i*5000
                max_value = min_value + 5000
                filtered_pair_concat = pair_concat[(pair_concat['Distance'] > min_value) & (pair_concat['Distance'] <= max_value)]
                filtered_window = window[(window['Distance'] > min_value) & (window['Distance'] <= max_value)]
                filtered_multicell = multicell[(multicell['Distance'] > min_value) & (multicell['Distance'] <= max_value)]
                correlation_pair_concat = filtered_pair_concat['TrueValue'].corr(filtered_pair_concat['PredictedValue'], method='pearson')
                correlation_window = filtered_window['TrueValue'].corr(filtered_window['PredictedValue'], method='pearson')
                correlation_multicell = filtered_multicell['TrueValue'].corr(filtered_multicell['PredictedValue'], method='pearson')
                pearson_array_pair_concat[i] = correlation_pair_concat
                pearson_array_window[i] = correlation_window
                pearson_array_multicell[i] = correlation_multicell

            # Plot on each subplot
            # Plot the three arrays
            axes[row][column].plot(x, pearson_array_pair_concat, label="PAIR-CONCAT", color="purple")
            axes[row][column].plot(x, pearson_array_window, label="WINDOW", color="cyan")
            axes[row][column].plot(x, pearson_array_multicell, label="MULTICELL", color="red")

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
    fig.suptitle("Train/test Cross-cell, on chr17")
    

    # Create custom legend handles
    purple_patch = mpatches.Patch(color='purple', label='PAIR-CONCAT')
    cyan_patch = mpatches.Patch(color='cyan', label='WINDOW')
    red_patch = mpatches.Patch(color='red', label='MULTICELL')
    
    # Add the custom legend to the figure
    fig.legend(
        handles=[purple_patch, cyan_patch, red_patch],
        loc="upper right",      # Top right of entire figure
        fontsize=8,
        bbox_to_anchor=(1, 1),  # Adjust if needed
    )

    # Adjust layout to ensure no overlap
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    
    save_path = "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_plots/pearson_plot_diff_cells_chr17.png"
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High-resolution save

    print(f"Plot saved at: {save_path}")






# Wrapper function to execute tasks
def run_task(task):
    func, *args = task
    func(*args)


# don't need to modify multicell function because there are already preprocessed files in folder
def main():
    import multiprocessing
    # Define the function and arguments for each task
    # train file, test file, path for final file
    tasks = [
        # use window files ('CV')
        (crossCell_pair_concat, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/WINDOW/chr17/CV/Gm12878_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/WINDOW/chr17/CV/K562_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Gm12878_K562_pc_chr17.csv"),
        (crossCell_pair_concat, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/WINDOW/chr17/CV/Gm12878_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/WINDOW/chr17/CV/Huvec_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Gm12878_Huvec_pc_chr17.csv"),
        (crossCell_pair_concat, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/WINDOW/chr17/CV/Gm12878_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/WINDOW/chr17/CV/Hmec_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Gm12878_Hmec_pc_chr17.csv"),
        (crossCell_pair_concat, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/WINDOW/chr17/CV/Gm12878_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/WINDOW/chr17/CV/Nhek_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Gm12878_Nhek_pc_chr17.csv"),

        (crossCell_pair_concat, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/WINDOW/chr17/CV/K562_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/WINDOW/chr17/CV/Gm12878_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_K562_Gm12878_pc_chr17.csv"),
        (crossCell_pair_concat, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/WINDOW/chr17/CV/K562_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/WINDOW/chr17/CV/Huvec_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_K562_Huvec_pc_chr17.csv"),
        (crossCell_pair_concat, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/WINDOW/chr17/CV/K562_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/WINDOW/chr17/CV/Hmec_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_K562_Hmec_pc_chr17.csv"),
        (crossCell_pair_concat, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/WINDOW/chr17/CV/K562_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/WINDOW/chr17/CV/Nhek_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_K562_Nhek_pc_chr17.csv"),

        (crossCell_pair_concat, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/WINDOW/chr17/CV/Huvec_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/WINDOW/chr17/CV/Gm12878_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Huvec_Gm12878_pc_chr17.csv"),
        (crossCell_pair_concat, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/WINDOW/chr17/CV/Huvec_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/WINDOW/chr17/CV/K562_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Huvec_K562_pc_chr17.csv"),
        (crossCell_pair_concat, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/WINDOW/chr17/CV/Huvec_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/WINDOW/chr17/CV/Hmec_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Huvec_Hmec_pc_chr17.csv"),
        (crossCell_pair_concat, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/WINDOW/chr17/CV/Huvec_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/WINDOW/chr17/CV/Nhek_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Huvec_Nhek_pc_chr17.csv"),

        (crossCell_pair_concat, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/WINDOW/chr17/CV/Hmec_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/WINDOW/chr17/CV/Gm12878_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Hmec_Gm12878_pc_chr17.csv"),
        (crossCell_pair_concat, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/WINDOW/chr17/CV/Hmec_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/WINDOW/chr17/CV/K562_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Hmec_K562_pc_chr17.csv"),
        (crossCell_pair_concat, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/WINDOW/chr17/CV/Hmec_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/WINDOW/chr17/CV/Huvec_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Hmec_Huvec_pc_chr17.csv"),
        (crossCell_pair_concat, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/WINDOW/chr17/CV/Hmec_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/WINDOW/chr17/CV/Nhek_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Hmec_Nhek_pc_chr17.csv"),

        (crossCell_pair_concat, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/WINDOW/chr17/CV/Nhek_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/WINDOW/chr17/CV/Gm12878_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Nhek_Gm12878_pc_chr17.csv"),
        (crossCell_pair_concat, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/WINDOW/chr17/CV/Nhek_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/WINDOW/chr17/CV/K562_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Nhek_K562_pc_chr17.csv"),
        (crossCell_pair_concat, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/WINDOW/chr17/CV/Nhek_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/WINDOW/chr17/CV/Huvec_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Nhek_Huvec_pc_chr17.csv"),
        (crossCell_pair_concat, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/WINDOW/chr17/CV/Nhek_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/WINDOW/chr17/CV/Hmec_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Nhek_Hmec_pc_chr17.csv"),


        # use window files ('CV')
        (crossCell_window, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/WINDOW/chr17/CV/Gm12878_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/WINDOW/chr17/CV/K562_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Gm12878_K562_w_chr17.csv"),
        (crossCell_window, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/WINDOW/chr17/CV/Gm12878_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/WINDOW/chr17/CV/Huvec_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Gm12878_Huvec_w_chr17.csv"),
        (crossCell_window, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/WINDOW/chr17/CV/Gm12878_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/WINDOW/chr17/CV/Hmec_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Gm12878_Hmec_w_chr17.csv"),
        (crossCell_window, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/WINDOW/chr17/CV/Gm12878_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/WINDOW/chr17/CV/Nhek_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Gm12878_Nhek_w_chr17.csv"),

        (crossCell_window, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/WINDOW/chr17/CV/K562_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/WINDOW/chr17/CV/Gm12878_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_K562_Gm12878_w_chr17.csv"),
        (crossCell_window, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/WINDOW/chr17/CV/K562_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/WINDOW/chr17/CV/Huvec_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_K562_Huvec_w_chr17.csv"),
        (crossCell_window, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/WINDOW/chr17/CV/K562_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/WINDOW/chr17/CV/Hmec_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_K562_Hmec_w_chr17.csv"),
        (crossCell_window, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/WINDOW/chr17/CV/K562_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/WINDOW/chr17/CV/Nhek_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_K562_Nhek_w_chr17.csv"),

        (crossCell_window, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/WINDOW/chr17/CV/Huvec_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/WINDOW/chr17/CV/Gm12878_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Huvec_Gm12878_w_chr17.csv"),
        (crossCell_window, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/WINDOW/chr17/CV/Huvec_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/WINDOW/chr17/CV/K562_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Huvec_K562_w_chr17.csv"),
        (crossCell_window, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/WINDOW/chr17/CV/Huvec_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/WINDOW/chr17/CV/Hmec_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Huvec_Hmec_w_chr17.csv"),
        (crossCell_window, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/WINDOW/chr17/CV/Huvec_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/WINDOW/chr17/CV/Nhek_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Huvec_Nhek_w_chr17.csv"),

        (crossCell_window, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/WINDOW/chr17/CV/Hmec_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/WINDOW/chr17/CV/Gm12878_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Hmec_Gm12878_w_chr17.csv"),
        (crossCell_window, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/WINDOW/chr17/CV/Hmec_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/WINDOW/chr17/CV/K562_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Hmec_K562_w_chr17.csv"),
        (crossCell_window, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/WINDOW/chr17/CV/Hmec_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/WINDOW/chr17/CV/Huvec_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Hmec_Huvec_w_chr17.csv"),
        (crossCell_window, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/WINDOW/chr17/CV/Hmec_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/WINDOW/chr17/CV/Nhek_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Hmec_Nhek_w_chr17.csv"),

        (crossCell_window, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/WINDOW/chr17/CV/Nhek_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/WINDOW/chr17/CV/Gm12878_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Nhek_Gm12878_w_chr17.csv"),
        (crossCell_window, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/WINDOW/chr17/CV/Nhek_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/WINDOW/chr17/CV/K562_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Nhek_K562_w_chr17.csv"),
        (crossCell_window, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/WINDOW/chr17/CV/Nhek_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/WINDOW/chr17/CV/Huvec_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Nhek_Huvec_w_chr17.csv"),
        (crossCell_window, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/WINDOW/chr17/CV/Nhek_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/WINDOW/chr17/CV/Hmec_chr17_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Nhek_Hmec_w_chr17.csv"),


        # use special multicell files provided (crosscell)
        (crossCell_multicell, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/MULTICELL/chr17/CrossCell/Gm12878toK562/Gm12878_chr17_train_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/MULTICELL/chr17/CrossCell/Gm12878toK562/K562_chr17_test_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Gm12878_K562_mc_chr17.csv"),
        (crossCell_multicell, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/MULTICELL/chr17/CrossCell/Gm12878toHuvec/Gm12878_chr17_train_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/MULTICELL/chr17/CrossCell/Gm12878toHuvec/Huvec_chr17_test_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Gm12878_Huvec_mc_chr17.csv"),
        (crossCell_multicell, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/MULTICELL/chr17/CrossCell/Gm12878toHmec/Gm12878_chr17_train_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/MULTICELL/chr17/CrossCell/Gm12878toHmec/Hmec_chr17_test_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Gm12878_Hmec_mc_chr17.csv"),
        (crossCell_multicell, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/MULTICELL/chr17/CrossCell/Gm12878toNhek/Gm12878_chr17_train_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/MULTICELL/chr17/CrossCell/Gm12878toNhek/Nhek_chr17_test_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Gm12878_Nhek_mc_chr17.csv"),

        (crossCell_multicell, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/MULTICELL/chr17/CrossCell/K562toGm12878/K562_chr17_train_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/MULTICELL/chr17/CrossCell/K562toGm12878/Gm12878_chr17_test_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_K562_Gm12878_mc_chr17.csv"),
        (crossCell_multicell, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/MULTICELL/chr17/CrossCell/K562toHuvec/K562_chr17_train_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/MULTICELL/chr17/CrossCell/K562toHuvec/Huvec_chr17_test_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_K562_Huvec_mc_chr17.csv"),
        (crossCell_multicell, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/MULTICELL/chr17/CrossCell/K562toHmec/K562_chr17_train_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/MULTICELL/chr17/CrossCell/K562toHmec/Hmec_chr17_test_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_K562_Hmec_mc_chr17.csv"),
        (crossCell_multicell, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/MULTICELL/chr17/CrossCell/K562toNhek/K562_chr17_train_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/MULTICELL/chr17/CrossCell/K562toNhek/Nhek_chr17_test_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_K562_Nhek_mc_chr17.csv"),

        (crossCell_multicell, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/MULTICELL/chr17/CrossCell/HuvectoGm12878/Huvec_chr17_train_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/MULTICELL/chr17/CrossCell/HuvectoGm12878/Gm12878_chr17_test_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Huvec_Gm12878_mc_chr17.csv"),
        (crossCell_multicell, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/MULTICELL/chr17/CrossCell/HuvectoK562/Huvec_chr17_train_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/MULTICELL/chr17/CrossCell/HuvectoK562/K562_chr17_test_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Huvec_K562_mc_chr17.csv"),
        (crossCell_multicell, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/MULTICELL/chr17/CrossCell/HuvectoHmec/Huvec_chr17_train_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/MULTICELL/chr17/CrossCell/HuvectoHmec/Hmec_chr17_test_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Huvec_Hmec_mc_chr17.csv"),
        (crossCell_multicell, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/MULTICELL/chr17/CrossCell/HuvectoNhek/Huvec_chr17_train_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/MULTICELL/chr17/CrossCell/HuvectoNhek/Nhek_chr17_test_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Huvec_Nhek_mc_chr17.csv"),

        (crossCell_multicell, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/MULTICELL/chr17/CrossCell/HmectoGm12878/Hmec_chr17_train_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/MULTICELL/chr17/CrossCell/HmectoGm12878/Gm12878_chr17_test_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Hmec_Gm12878_mc_chr17.csv"),
        (crossCell_multicell, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/MULTICELL/chr17/CrossCell/HmectoK562/Hmec_chr17_train_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/MULTICELL/chr17/CrossCell/HmectoK562/K562_chr17_test_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Hmec_K562_mc_chr17.csv"),
        (crossCell_multicell, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/MULTICELL/chr17/CrossCell/HmectoHuvec/Hmec_chr17_train_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/MULTICELL/chr17/CrossCell/HmectoHuvec/Huvec_chr17_test_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Hmec_Huvec_mc_chr17.csv"),
        (crossCell_multicell, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/MULTICELL/chr17/CrossCell/HmectoNhek/Hmec_chr17_train_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/MULTICELL/chr17/CrossCell/HmectoNhek/Nhek_chr17_test_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Hmec_Nhek_mc_chr17.csv"),

        (crossCell_multicell, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/MULTICELL/chr17/CrossCell/NhektoGm12878/Nhek_chr17_train_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/MULTICELL/chr17/CrossCell/NhektoGm12878/Gm12878_chr17_test_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Nhek_Gm12878_mc_chr17.csv"),
        (crossCell_multicell, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/MULTICELL/chr17/CrossCell/NhektoK562/Nhek_chr17_train_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/MULTICELL/chr17/CrossCell/NhektoK562/K562_chr17_test_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Nhek_K562_mc_chr17.csv"),
        (crossCell_multicell, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/MULTICELL/chr17/CrossCell/NhektoHuvec/Nhek_chr17_train_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/MULTICELL/chr17/CrossCell/NhektoHuvec/Huvec_chr17_test_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Nhek_Huvec_mc_chr17.csv"),
        (crossCell_multicell, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/MULTICELL/chr17/CrossCell/NhektoHmec/Nhek_chr17_train_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/MULTICELL/chr17/CrossCell/NhektoHmec/Hmec_chr17_test_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_cell_predictions/predictions_Nhek_Hmec_mc_chr17.csv")
    ]

    
    with multiprocessing.Pool(processes=len(tasks)) as pool:  # Use all available cores
        pool.map(run_task, tasks)
    
    # make 1 plot containing 5*5 subplots
    pearson_plots()

if __name__ == '__main__':
    main()

