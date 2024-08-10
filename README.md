README: Neuronal Waveform Clustering and Analysis

link to final report: https://github.com/neoygaj/CSPB-4502-project/blob/main/Group4_NeuropixelsSC_part6.pdf
link to presentation video: could not upload to GitHub due to 25mb file size limit, no matter how low the resolution or zipped.  Messaged Instructor prior to submission deadline

Overview

This MATLAB script is designed to analyze neuronal waveforms recorded from the superior colliculus (SC) of live mice in response to stimuli that elicit saccades. The script processes waveform data and direction selectivity measurements to perform clustering analysis, identify key attributes, and generate visualizations of the clustering results.

Files and Input Data

spikewaveforms: A numerical matrix containing the waveform data for all recorded neurons. This data should be imported as a text file.

DS: A numerical matrix containing direction selectivity measurements for each neuron. This data should also be imported as a text file.

Script Workflow

1. Data Import and Inspection
The script begins by importing the waveform and direction selectivity data. The waveform data is transposed for easier manipulation and plotted to provide an initial visual inspection of all waveforms across neurons.

2. Data Cleaning
The script includes a loop that checks for and counts any NaN values or zeros in the waveform data and direction selectivity measurements. Detected NaNs are converted to zeros to ensure clean data for further analysis.

3. Attribute Calculation
For each waveform, the script calculates key attributes:
MAX: Maximum amplitude.
MIN: Minimum amplitude.
AUC: Area under the curve, calculated using the trapezoidal rule.
The script categorizes neurons as "ON" or "OFF" based on the waveform attributes.

4. Peristimulus Activity Focus
The script focuses on timepoints 10-40 to analyze peristimulus activity, identifying neurons with significant activity within this time window.

5. Clustering Analysis
Hierarchical Clustering: The script performs hierarchical clustering using Ward’s method, followed by PCA for dimensionality reduction. The results are visualized with a scatter plot.
K-means Clustering: The script also performs K-means clustering with 8 clusters and visualizes the clustering results using PCA.

6. Waveform Visualization
Separate plots for each cluster are generated to visualize the waveforms grouped by K-means and hierarchical clustering methods.

7. Scatter Plots and Linear Models
Scatter plots are created to explore relationships between waveform attributes (e.g., min amplitude vs. AUC). Linear models are fitted to these relationships to assess potential correlations.
Results

The script generates a series of figures that provide insights into the clustering and relationships between waveform attributes. These include:
Waveform clusters for both K-means and hierarchical methods.
PCA scatter plots for each clustering method.
Scatter plots comparing various waveform attributes, with fitted linear models.

How to Run

Import the data: Ensure the waveform data (spikewaveforms) and direction selectivity measurements (DS) are correctly imported as numerical matrices.
Run the script: Execute the script in MATLAB to perform the clustering analysis and generate the figures.

Inspect the results: Review the generated plots and scatter plots to interpret the clustering results and relationships between waveform attributes.

Notes

The script uses parallel processing (via parfor) to speed up the computation of certain loops. Ensure that MATLAB’s Parallel Computing Toolbox is installed and configured if parallel processing is enabled.
The number of clusters for K-means and hierarchical clustering can be adjusted by modifying the numClusters variable.
