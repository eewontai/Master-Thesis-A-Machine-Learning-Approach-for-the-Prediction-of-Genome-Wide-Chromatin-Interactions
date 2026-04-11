#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 12:52:47 2025

@author: eta

Validate the results of the paper - make cross-chromosome analysis and plots
Train on chromosome 17, testing on chromosome 14
"""

# get text files of chr17 and chr14 features for one cell line (authors' dataset), 
# train on the entire dataset of chr17, test on the entire dataset of chr14, 
# save prediction file in path
# path: complete path to the file that will be saved as predictions, including filename.csv, in string form
# pair-concat: use window files from authors and delete the window features
def train17_test14_pair_concat(file_train, file_test, path):
    # import libraries
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    # read text files and make dataframes
    # drop window features
    df_train = pd.read_csv(file_train, sep='\s+').drop(['Ctcf_W', 'Dnase_W', 'H3k27ac_W', 'H3k27me3_W', 'H3k36me3_W', 'H3k4me1_W', 'H3k4me2_W', 'H3k4me3_W', 'H3k79me2_W', 'H3k9ac_W', 'H3k9me3_W', 'H4k20me1_W', 'RAD21_W', 'TBP_W'], axis=1)
    df_test = pd.read_csv(file_test, sep='\s+').drop(['Ctcf_W', 'Dnase_W', 'H3k27ac_W', 'H3k27me3_W', 'H3k36me3_W', 'H3k4me1_W', 'H3k4me2_W', 'H3k4me3_W', 'H3k79me2_W', 'H3k9ac_W', 'H3k9me3_W', 'H4k20me1_W', 'RAD21_W', 'TBP_W'], axis=1)

    # make train and test sets
    # X_train and X_test should exclude the first column, which contains information about the genomic location, which is irrelavant to model training
    # y_train and y_test should only include the Count column, which is the HiC count values
    X_train = df_train.iloc[:, 1:(len(df_train.columns)-1)]
    y_train = df_train[['Count']]
    X_test = df_test.iloc[:, 1:(len(df_test.columns)-1)]
    y_test = df_test[['Count']]

    # scale and normalize columns in X, since Distance column has very high values compared to other columns
    # Initialize the scaler
    scaler = StandardScaler()
    scaler.fit(X_train)  # Fit on train set only

    # Transform train and test sets using the scaler
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Do not fit again!
    
    # Transform y from column vector to 1D array
    y_train_array = y_train.to_numpy().flatten()
    y_test_array = y_test.to_numpy().flatten()

    # Define the model - scikit learn random forest regression
    # n_estimators: default=100. the number of trees in the forest
    model = RandomForestRegressor(n_estimators=20, random_state=42)
    
    # train model on training set
    model.fit(X_train_scaled, y_train_array)
    
    # make predictions dataframe, consisting of genomic region pair, true count value, model prediction, and distance between the two regions
    predictions = pd.DataFrame({
        'Pair': df_test['Pair'],
        'TrueValue': y_test_array,
        'PredictedValue': model.predict(X_test_scaled),
        'Distance': df_test['Distance']
        })
    

    # save predictions to a file
    predictions.to_csv(path, index=False)



# get text files of chr17 and chr14 features for one cell line (authors' dataset), 
# train on the entire dataset of chr17, test on the entire dataset of chr14, 
# save prediction file in path
# path: complete path to the file that will be saved as predictions, including filename.csv, in string form
# window: use window files from authors
def train17_test14_window(file_train, file_test, path):
    # import libraries
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    
    # read text files and make dataframes
    df_train = pd.read_csv(file_train, sep='\s+')
    df_test = pd.read_csv(file_test, sep='\s+')
    
    # make train and test sets
    # X_train and X_test should exclude the first column, which contains information about the genomic location, which is irrelavant to model training
    # y_train and y_test should only include the Count column, which is the HiC count values
    X_train = df_train.iloc[:, 1:(len(df_train.columns)-1)]
    y_train = df_train[['Count']]
    X_test = df_test.iloc[:, 1:(len(df_test.columns)-1)]
    y_test = df_test[['Count']]

    # scale and normalize columns in X, since Distance column has very high values compared to other columns
    # Initialize the scaler
    scaler = StandardScaler()
    scaler.fit(X_train)  # Fit on train set only

    # Transform train and test sets using the scaler
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Do not fit again!

    # Transform y from column vector to 1D array
    y_train_array = y_train.to_numpy().flatten()
    y_test_array = y_test.to_numpy().flatten()

    # Define the model - scikit learn random forest regression
    model = RandomForestRegressor(n_estimators=20, random_state=42)
    # n_estimators: default=100. the number of trees in the forest

    # train model on training set
    model.fit(X_train_scaled, y_train_array)
    
    # make predictions dataframe, consisting of genomic region pair, true count value, model prediction, and distance between the two regions
    predictions = pd.DataFrame({
        'Pair': df_test['Pair'],
        'TrueValue': y_test_array,
        'PredictedValue': model.predict(X_test_scaled),
        'Distance': df_test['Distance']
        })

    # save predictions to a file
    predictions.to_csv(path, index=False)




# get text files of chr17 and chr14 features for one cell line (authors' dataset), 
# train on the entire dataset of chr17, test on the entire dataset of chr14, 
# save prediction file in path
# path: complete path to the file that will be saved as predictions, including filename.csv, in string form
# multicell: use multicell files from authors
def train17_test14_multicell(file_train, file_test, path):
    # import libraries
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    
    # read text files and make dataframes
    df_train = pd.read_csv(file_train, sep='\s+')
    df_test = pd.read_csv(file_test, sep='\s+')

    # make train and test sets
    # X_train and X_test should exclude the first column, which contains information about the genomic location, which is irrelavant to model training
    # y_train and y_test should only include the Count column, which is the HiC count values
    X_train = df_train.iloc[:, 1:(len(df_train.columns)-1)]
    y_train = df_train[['Count']]
    X_test = df_test.iloc[:, 1:(len(df_test.columns)-1)]
    y_test = df_test[['Count']]

    # scale and normalize columns in X, since Distance column has very high values compared to other columns
    # Initialize the scaler
    scaler = StandardScaler()
    scaler.fit(X_train)  # Fit on train set only

    # Transform train and test sets using the scaler
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Do not fit again!

    # Transform y from column vector to 1D array
    y_train_array = y_train.to_numpy().flatten()
    y_test_array = y_test.to_numpy().flatten()

    # Define the model - scikit learn random forest regression
    model = RandomForestRegressor(n_estimators=20, random_state=42)
    # n_estimators: default=100. the number of trees in the forest

    # train model on training set
    model.fit(X_train_scaled, y_train_array)
    
    # make predictions dataframe, consisting of genomic region pair, true count value, model prediction, and distance between the two regions
    predictions = pd.DataFrame({
        'Pair': df_test['Pair'],
        'TrueValue': y_test_array,
        'PredictedValue': model.predict(X_test_scaled),
        'Distance': df_test['Distance']
        })

    # save predictions to a file
    predictions.to_csv(path, index=False)




# generate distance-stratified pearson's correlation plots from the prediction files generated above,
# including three lines - pair-concat, window and multicell - in one plot
# for cell line gm12878
def pearson_plot_gm12878():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    # read prediction files and make them into dataframes
    gm12878_pair_concat = pd.read_csv("/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_predictions/Gm12878_pair_concat_predictions_train17_test14_1st.csv")
    gm12878_window = pd.read_csv("/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_predictions/Gm12878_window_predictions_train17_test14_1st.csv")
    gm12878_multicell = pd.read_csv("/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_predictions/Gm12878_multicell_predictions_train17_test14_1st.csv")
    
    
    # make bins of Distance values (from 0Mbp to 1Mbp)
    # total 200 bins of size 5kbp
    # >0, <=5000
    # >5000, <=10,000
    # ...
    # >995,000 , <=1,000,000
    
    # make numpy arrays
    pearson_array_gm12878_pair_concat = np.zeros(200)
    pearson_array_gm12878_window = np.zeros(200)
    pearson_array_gm12878_multicell = np.zeros(200)
    
    # for each distance bin, calculate pearson's correlation between true value and predicted value of HiC count
    for i in range(200):
        # start and end location of the 5kb bin
        min_value = i*5000
        max_value = min_value + 5000
        # make group that it's distance value is within the distance bin
        # for pair-concat, window and multicell
        filtered_gm12878_pair_concat = gm12878_pair_concat[(gm12878_pair_concat['Distance'] > min_value) & (gm12878_pair_concat['Distance'] <= max_value)]
        filtered_gm12878_window = gm12878_window[(gm12878_window['Distance'] > min_value) & (gm12878_window['Distance'] <= max_value)]
        filtered_gm12878_multicell = gm12878_multicell[(gm12878_multicell['Distance'] > min_value) & (gm12878_multicell['Distance'] <= max_value)]
        # for each distance bin, make pearson's correltion coefficient between truevalue and predictedvalue
        # for pair-concat, window and multicell
        correlation_gm12878_pair_concat = filtered_gm12878_pair_concat['TrueValue'].corr(filtered_gm12878_pair_concat['PredictedValue'], method='pearson')
        correlation_gm12878_window = filtered_gm12878_window['TrueValue'].corr(filtered_gm12878_window['PredictedValue'], method='pearson')
        correlation_gm12878_multicell = filtered_gm12878_multicell['TrueValue'].corr(filtered_gm12878_multicell['PredictedValue'], method='pearson')
        # get the resulting value into a value in an array
        pearson_array_gm12878_pair_concat[i] = correlation_gm12878_pair_concat
        pearson_array_gm12878_window[i] = correlation_gm12878_window
        pearson_array_gm12878_multicell[i] = correlation_gm12878_multicell
    
    # plotting - 3 lines in each plot, pair-concat, window and multicell
    # X-axis: indices (0 to 199)
    x = np.arange(200)
    x = np.array(x) / 200  # Normalize x values to range [0,1]
    # Plot the three arrays
    plt.figure(figsize=(5, 5))  # Set figure size
    plt.plot(x, pearson_array_gm12878_pair_concat, label="PAIR-CONCAT", linestyle="-", marker="o", markersize=3, color="green")
    plt.plot(x, pearson_array_gm12878_window, label="WINDOW", linestyle="--", marker="s", markersize=3, color="cyan")
    plt.plot(x, pearson_array_gm12878_multicell, label="MULTICELL", linestyle="-.", marker="d", markersize=3, color="red")
    
    # Labels and title
    plt.xlabel("Distance (Mb)")
    plt.ylabel("Correlation")
    plt.title("Gm12878 - train on chr17, test on chr14")
    plt.legend()  # Show legend
    
    # Set axis limits to [0,1]
    plt.xlim(0, 1)
    plt.ylim(-0.2, 0.8)
    
    # Set ticks and grid
    plt.xticks([0, 0.5, 1])
    plt.yticks([-0.2, 0, 0.2, 0.4, 0.6])
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Specify folder and filename
    save_folder = "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_plots"
    save_path = os.path.join(save_folder, "Gm12878_train17_test14_RF_pearson.png")  # Full file path
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High-resolution save
    
    print(f"Plot saved at: {save_path}")


# generate distance-stratified pearson's correlation plots from the prediction files generated above,
# including three lines - pair-concat, window and multicell - in one plot
# for cell line hmec
def pearson_plot_hmec():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    # read prediction files and make them into dataframes
    hmec_pair_concat = pd.read_csv("/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_predictions/Hmec_pair_concat_predictions_train17_test14_1st.csv")
    hmec_window = pd.read_csv("/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_predictions/Hmec_window_predictions_train17_test14_1st.csv")
    hmec_multicell = pd.read_csv("/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_predictions/Hmec_multicell_predictions_train17_test14_1st.csv")
    
    
    # make bins of Distance values (from 0Mbp to 1Mbp)
    # total 200 bins of size 5kbp
    # >0, <=5000
    # >5000, <=10,000
    # ...
    # >995,000 , <=1,000,000
    
    # make numpy arrays
    pearson_array_hmec_pair_concat = np.zeros(200)
    pearson_array_hmec_window = np.zeros(200)
    pearson_array_hmec_multicell = np.zeros(200)
    
    # for each distance bin, calculate pearson's correlation between true value and predicted value of HiC count
    for i in range(200):
        # start and end location of the 5kb bin
        min_value = i*5000
        max_value = min_value + 5000
        # make group that it's distance value is within the distance bin
        # for pair-concat, window and multicell
        filtered_hmec_pair_concat = hmec_pair_concat[(hmec_pair_concat['Distance'] > min_value) & (hmec_pair_concat['Distance'] <= max_value)]
        filtered_hmec_window = hmec_window[(hmec_window['Distance'] > min_value) & (hmec_window['Distance'] <= max_value)]
        filtered_hmec_multicell = hmec_multicell[(hmec_multicell['Distance'] > min_value) & (hmec_multicell['Distance'] <= max_value)]
        # for each distance bin, make pearson's correltion coefficient between truevalue and predictedvalue
        # for pair-concat, window and multicell
        correlation_hmec_pair_concat = filtered_hmec_pair_concat['TrueValue'].corr(filtered_hmec_pair_concat['PredictedValue'], method='pearson')
        correlation_hmec_window = filtered_hmec_window['TrueValue'].corr(filtered_hmec_window['PredictedValue'], method='pearson')
        correlation_hmec_multicell = filtered_hmec_multicell['TrueValue'].corr(filtered_hmec_multicell['PredictedValue'], method='pearson')
        # get the resulting value into a value in an array
        pearson_array_hmec_pair_concat[i] = correlation_hmec_pair_concat
        pearson_array_hmec_window[i] = correlation_hmec_window
        pearson_array_hmec_multicell[i] = correlation_hmec_multicell
    
    # plotting - 3 lines in each plot, pair-concat, window and multicell
    # X-axis: indices (0 to 199)
    x = np.arange(200)
    x = np.array(x) / 200  # Normalize x values to range [0,1]
    # Plot the three arrays
    plt.figure(figsize=(5, 5))  # Set figure size
    plt.plot(x, pearson_array_hmec_pair_concat, label="PAIR-CONCAT", linestyle="-", marker="o", markersize=3, color="green")
    plt.plot(x, pearson_array_hmec_window, label="WINDOW", linestyle="--", marker="s", markersize=3, color="cyan")
    plt.plot(x, pearson_array_hmec_multicell, label="MULTICELL", linestyle="-.", marker="d", markersize=3, color="red")
    
    # Labels and title
    plt.xlabel("Distance (Mb)")
    plt.ylabel("Correlation")
    plt.title("Hmec - train on chr17, test on chr14")
    plt.legend()  # Show legend
    
    # Set axis limits to [0,1]
    plt.xlim(0, 1)
    plt.ylim(-0.2, 0.8)
    
    # Set ticks and grid
    plt.xticks([0, 0.5, 1])
    plt.yticks([-0.2, 0, 0.2, 0.4, 0.6])
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Specify folder and filename
    save_folder = "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_plots"
    save_path = os.path.join(save_folder, "Hmec_train17_test14_RF_pearson.png")  # Full file path
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High-resolution save
    
    print(f"Plot saved at: {save_path}")
    


# generate distance-stratified pearson's correlation plots from the prediction files generated above,
# including three lines - pair-concat, window and multicell - in one plot
# for cell line huvec
def pearson_plot_huvec():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    # read prediction files and make them into dataframes
    huvec_pair_concat = pd.read_csv("/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_predictions/Huvec_pair_concat_predictions_train17_test14_1st.csv")
    huvec_window = pd.read_csv("/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_predictions/Huvec_window_predictions_train17_test14_1st.csv")
    huvec_multicell = pd.read_csv("/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_predictions/Huvec_multicell_predictions_train17_test14_1st.csv")
    
    
    # make bins of Distance values (from 0Mbp to 1Mbp)
    # total 200 bins of size 5kbp
    # >0, <=5000
    # >5000, <=10,000
    # ...
    # >995,000 , <=1,000,000
    
    # make numpy arrays
    pearson_array_huvec_pair_concat = np.zeros(200)
    pearson_array_huvec_window = np.zeros(200)
    pearson_array_huvec_multicell = np.zeros(200)
    
    # for each distance bin, calculate pearson's correlation between true value and predicted value of HiC count
    for i in range(200):
        # start and end location of the 5kb bin
        min_value = i*5000
        max_value = min_value + 5000
        # make group that it's distance value is within the distance bin
        # for pair-concat, window and multicell
        filtered_huvec_pair_concat = huvec_pair_concat[(huvec_pair_concat['Distance'] > min_value) & (huvec_pair_concat['Distance'] <= max_value)]
        filtered_huvec_window = huvec_window[(huvec_window['Distance'] > min_value) & (huvec_window['Distance'] <= max_value)]
        filtered_huvec_multicell = huvec_multicell[(huvec_multicell['Distance'] > min_value) & (huvec_multicell['Distance'] <= max_value)]
        # for each distance bin, make pearson's correltion coefficient between truevalue and predictedvalue
        # for pair-concat, window and multicell
        correlation_huvec_pair_concat = filtered_huvec_pair_concat['TrueValue'].corr(filtered_huvec_pair_concat['PredictedValue'], method='pearson')
        correlation_huvec_window = filtered_huvec_window['TrueValue'].corr(filtered_huvec_window['PredictedValue'], method='pearson')
        correlation_huvec_multicell = filtered_huvec_multicell['TrueValue'].corr(filtered_huvec_multicell['PredictedValue'], method='pearson')
        # get the resulting value into a value in an array
        pearson_array_huvec_pair_concat[i] = correlation_huvec_pair_concat
        pearson_array_huvec_window[i] = correlation_huvec_window
        pearson_array_huvec_multicell[i] = correlation_huvec_multicell
    
    # plotting - 3 lines in each plot, pair-concat, window and multicell
    # X-axis: indices (0 to 199)
    x = np.arange(200)
    x = np.array(x) / 200  # Normalize x values to range [0,1]
    # Plot the three arrays
    plt.figure(figsize=(5, 5))  # Set figure size
    plt.plot(x, pearson_array_huvec_pair_concat, label="PAIR-CONCAT", linestyle="-", marker="o", markersize=3, color="green")
    plt.plot(x, pearson_array_huvec_window, label="WINDOW", linestyle="--", marker="s", markersize=3, color="cyan")
    plt.plot(x, pearson_array_huvec_multicell, label="MULTICELL", linestyle="-.", marker="d", markersize=3, color="red")
    
    # Labels and title
    plt.xlabel("Distance (Mb)")
    plt.ylabel("Correlation")
    plt.title("Huvec - train on chr17, test on chr14")
    plt.legend()  # Show legend
    
    # Set axis limits to [0,1]
    plt.xlim(0, 1)
    plt.ylim(-0.2, 0.8)
    
    # Set ticks and grid
    plt.xticks([0, 0.5, 1])
    plt.yticks([-0.2, 0, 0.2, 0.4, 0.6])
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Specify folder and filename
    save_folder = "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_plots"
    save_path = os.path.join(save_folder, "Huvec_train17_test14_RF_pearson.png")  # Full file path
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High-resolution save
    
    print(f"Plot saved at: {save_path}")
    
    

# generate distance-stratified pearson's correlation plots from the prediction files generated above,
# including three lines - pair-concat, window and multicell - in one plot
# for cell line k562
def pearson_plot_k562():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    # read prediction files and make them into dataframes
    k562_pair_concat = pd.read_csv("/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_predictions/K562_pair_concat_predictions_train17_test14_1st.csv")
    k562_window = pd.read_csv("/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_predictions/K562_window_predictions_train17_test14_1st.csv")
    k562_multicell = pd.read_csv("/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_predictions/K562_multicell_predictions_train17_test14_1st.csv")
    
    
    # make bins of Distance values (from 0Mbp to 1Mbp)
    # total 200 bins of size 5kbp
    # >0, <=5000
    # >5000, <=10,000
    # ...
    # >995,000 , <=1,000,000
    
    # make numpy arrays
    pearson_array_k562_pair_concat = np.zeros(200)
    pearson_array_k562_window = np.zeros(200)
    pearson_array_k562_multicell = np.zeros(200)
    
    # for each distance bin, calculate pearson's correlation between true value and predicted value of HiC count
    for i in range(200):
        # start and end location of the 5kb bin
        min_value = i*5000
        max_value = min_value + 5000
        # make group that it's distance value is within the distance bin
        # for pair-concat, window and multicell
        filtered_k562_pair_concat = k562_pair_concat[(k562_pair_concat['Distance'] > min_value) & (k562_pair_concat['Distance'] <= max_value)]
        filtered_k562_window = k562_window[(k562_window['Distance'] > min_value) & (k562_window['Distance'] <= max_value)]
        filtered_k562_multicell = k562_multicell[(k562_multicell['Distance'] > min_value) & (k562_multicell['Distance'] <= max_value)]
        # for each distance bin, make pearson's correltion coefficient between truevalue and predictedvalue
        # for pair-concat, window and multicell
        correlation_k562_pair_concat = filtered_k562_pair_concat['TrueValue'].corr(filtered_k562_pair_concat['PredictedValue'], method='pearson')
        correlation_k562_window = filtered_k562_window['TrueValue'].corr(filtered_k562_window['PredictedValue'], method='pearson')
        correlation_k562_multicell = filtered_k562_multicell['TrueValue'].corr(filtered_k562_multicell['PredictedValue'], method='pearson')
        # get the resulting value into a value in an array
        pearson_array_k562_pair_concat[i] = correlation_k562_pair_concat
        pearson_array_k562_window[i] = correlation_k562_window
        pearson_array_k562_multicell[i] = correlation_k562_multicell
    
    # plotting - 3 lines in each plot, pair-concat, window and multicell
    # X-axis: indices (0 to 199)
    x = np.arange(200)
    x = np.array(x) / 200  # Normalize x values to range [0,1]
    # Plot the three arrays
    plt.figure(figsize=(5, 5))  # Set figure size
    plt.plot(x, pearson_array_k562_pair_concat, label="PAIR-CONCAT", linestyle="-", marker="o", markersize=3, color="green")
    plt.plot(x, pearson_array_k562_window, label="WINDOW", linestyle="--", marker="s", markersize=3, color="cyan")
    plt.plot(x, pearson_array_k562_multicell, label="MULTICELL", linestyle="-.", marker="d", markersize=3, color="red")
    
    # Labels and title
    plt.xlabel("Distance (Mb)")
    plt.ylabel("Correlation")
    plt.title("K562 - train on chr17, test on chr14")
    plt.legend()  # Show legend
    
    # Set axis limits to [0,1]
    plt.xlim(0, 1)
    plt.ylim(-0.2, 0.8)
    
    # Set ticks and grid
    plt.xticks([0, 0.5, 1])
    plt.yticks([-0.2, 0, 0.2, 0.4, 0.6])
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Specify folder and filename
    save_folder = "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_plots"
    save_path = os.path.join(save_folder, "K562_train17_test14_RF_pearson.png")  # Full file path
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High-resolution save
    
    print(f"Plot saved at: {save_path}")
    
    

    
# generate distance-stratified pearson's correlation plots from the prediction files generated above,
# including three lines - pair-concat, window and multicell - in one plot
# for cell line nhek
def pearson_plot_nhek():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    # read prediction files and make them into dataframes
    nhek_pair_concat = pd.read_csv("/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_predictions/Nhek_pair_concat_predictions_train17_test14_1st.csv")
    nhek_window = pd.read_csv("/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_predictions/Nhek_window_predictions_train17_test14_1st.csv")
    nhek_multicell = pd.read_csv("/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_predictions/Nhek_multicell_predictions_train17_test14_1st.csv")
    
    
    # make bins of Distance values (from 0Mbp to 1Mbp)
    # total 200 bins of size 5kbp
    # >0, <=5000
    # >5000, <=10,000
    # ...
    # >995,000 , <=1,000,000
    
    # make numpy arrays
    pearson_array_nhek_pair_concat = np.zeros(200)
    pearson_array_nhek_window = np.zeros(200)
    pearson_array_nhek_multicell = np.zeros(200)
    
    # for each distance bin, calculate pearson's correlation between true value and predicted value of HiC count
    for i in range(200):
        # start and end location of the 5kb bin
        min_value = i*5000
        max_value = min_value + 5000
        # make group that it's distance value is within the distance bin
        # for pair-concat, window and multicell
        filtered_nhek_pair_concat = nhek_pair_concat[(nhek_pair_concat['Distance'] > min_value) & (nhek_pair_concat['Distance'] <= max_value)]
        filtered_nhek_window = nhek_window[(nhek_window['Distance'] > min_value) & (nhek_window['Distance'] <= max_value)]
        filtered_nhek_multicell = nhek_multicell[(nhek_multicell['Distance'] > min_value) & (nhek_multicell['Distance'] <= max_value)]
        # for each distance bin, make pearson's correltion coefficient between truevalue and predictedvalue
        # for pair-concat, window and multicell
        correlation_nhek_pair_concat = filtered_nhek_pair_concat['TrueValue'].corr(filtered_nhek_pair_concat['PredictedValue'], method='pearson')
        correlation_nhek_window = filtered_nhek_window['TrueValue'].corr(filtered_nhek_window['PredictedValue'], method='pearson')
        correlation_nhek_multicell = filtered_nhek_multicell['TrueValue'].corr(filtered_nhek_multicell['PredictedValue'], method='pearson')
        # get the resulting value into a value in an array
        pearson_array_nhek_pair_concat[i] = correlation_nhek_pair_concat
        pearson_array_nhek_window[i] = correlation_nhek_window
        pearson_array_nhek_multicell[i] = correlation_nhek_multicell
    
    # plotting - 3 lines in each plot, pair-concat, window and multicell
    # X-axis: indices (0 to 199)
    x = np.arange(200)
    x = np.array(x) / 200  # Normalize x values to range [0,1]
    # Plot the three arrays
    plt.figure(figsize=(5, 5))  # Set figure size
    plt.plot(x, pearson_array_nhek_pair_concat, label="PAIR-CONCAT", linestyle="-", marker="o", markersize=3, color="green")
    plt.plot(x, pearson_array_nhek_window, label="WINDOW", linestyle="--", marker="s", markersize=3, color="cyan")
    plt.plot(x, pearson_array_nhek_multicell, label="MULTICELL", linestyle="-.", marker="d", markersize=3, color="red")
    
    # Labels and title
    plt.xlabel("Distance (Mb)")
    plt.ylabel("Correlation")
    plt.title("Nhek - train on chr17, test on chr14")
    plt.legend()  # Show legend
    
    # Set axis limits to [0,1]
    plt.xlim(0, 1)
    plt.ylim(-0.2, 0.8)
    
    # Set ticks and grid
    plt.xticks([0, 0.5, 1])
    plt.yticks([-0.2, 0, 0.2, 0.4, 0.6])
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Specify folder and filename
    save_folder = "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_plots"
    save_path = os.path.join(save_folder, "Nhek_train17_test14_RF_pearson.png")  # Full file path
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High-resolution save
    
    print(f"Plot saved at: {save_path}")



# import library for parallel processing
import multiprocessing

# Define the function and arguments for each task
tasks = [
    (train17_test14_multicell, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/MULTICELL/chr17/CV/Gm12878_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/MULTICELL/chr14/CV/Gm12878_chr14_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_predictions/Gm12878_multicell_predictions_train17_test14_1st.csv"),
    (train17_test14_multicell, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/MULTICELL/chr17/CV/K562_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/MULTICELL/chr14/CV/K562_chr14_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_predictions/K562_multicell_predictions_train17_test14_1st.csv"),
    (train17_test14_multicell, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/MULTICELL/chr17/CV/Huvec_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/MULTICELL/chr14/CV/Huvec_chr14_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_predictions/Huvec_multicell_predictions_train17_test14_1st.csv"),
    (train17_test14_multicell, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/MULTICELL/chr17/CV/Hmec_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/MULTICELL/chr14/CV/Hmec_chr14_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_predictions/Hmec_multicell_predictions_train17_test14_1st.csv"),
    (train17_test14_multicell, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/MULTICELL/chr17/CV/Nhek_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/MULTICELL/chr14/CV/Nhek_chr14_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_predictions/Nhek_multicell_predictions_train17_test14_1st.csv"),
    
    (train17_test14_window, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/WINDOW/chr17/CV/Gm12878_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/WINDOW/chr14/CV/Gm12878_chr14_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_predictions/Gm12878_window_predictions_train17_test14_1st.csv"),
    (train17_test14_window, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/WINDOW/chr17/CV/K562_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/WINDOW/chr14/CV/K562_chr14_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_predictions/K562_window_predictions_train17_test14_1st.csv"),
    (train17_test14_window, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/WINDOW/chr17/CV/Huvec_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/WINDOW/chr14/CV/Huvec_chr14_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_predictions/Huvec_window_predictions_train17_test14_1st.csv"),
    (train17_test14_window, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/WINDOW/chr17/CV/Hmec_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/WINDOW/chr14/CV/Hmec_chr14_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_predictions/Hmec_window_predictions_train17_test14_1st.csv"),
    (train17_test14_window, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/WINDOW/chr17/CV/Nhek_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/WINDOW/chr14/CV/Nhek_chr14_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_predictions/Nhek_window_predictions_train17_test14_1st.csv"),

    (train17_test14_pair_concat, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/WINDOW/chr17/CV/Gm12878_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Gm12878/WINDOW/chr14/CV/Gm12878_chr14_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_predictions/Gm12878_pair_concat_predictions_train17_test14_1st.csv"),
    (train17_test14_pair_concat, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/WINDOW/chr17/CV/K562_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/K562/WINDOW/chr14/CV/K562_chr14_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_predictions/K562_pair_concat_predictions_train17_test14_1st.csv"),
    (train17_test14_pair_concat, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/WINDOW/chr17/CV/Huvec_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Huvec/WINDOW/chr14/CV/Huvec_chr14_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_predictions/Huvec_pair_concat_predictions_train17_test14_1st.csv"),
    (train17_test14_pair_concat, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/WINDOW/chr17/CV/Hmec_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Hmec/WINDOW/chr14/CV/Hmec_chr14_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_predictions/Hmec_pair_concat_predictions_train17_test14_1st.csv"),
    (train17_test14_pair_concat, "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/WINDOW/chr17/CV/Nhek_chr17_feature.txt", "/sybig/projects/GeneRegulation/data/Eewon/chromatin_interactions/source_files/Nhek/WINDOW/chr14/CV/Nhek_chr14_feature.txt", "/sybig/home/eta/Masterthesis/scripts/validate_paper_results/cross_chr_predictions/Nhek_pair_concat_predictions_train17_test14_1st.csv")
]

# Wrapper function to execute tasks
def run_task(task):
    func, *args = task
    func(*args)

def main():
    # Run tasks in parallel
    # make prediction files
    
    with multiprocessing.Pool(processes=len(tasks)) as pool:  # Use all available cores
        pool.map(run_task, tasks)
    
    # make plots
    pearson_plot_gm12878()
    pearson_plot_hmec()
    pearson_plot_huvec()
    pearson_plot_k562()
    pearson_plot_nhek()

if __name__ == "__main__":
    main()