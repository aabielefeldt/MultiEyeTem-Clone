# MultiEyeTem Analysis

This repository contains the analysis core for the MultiEyeTem project.

## Installation

To install a Python environment suitable to running this analysis code, please:

1. Download Miniconda for your system: https://docs.conda.io/en/latest/miniconda.html (this will provide you with the conda command)
2. Use conda to install mamba: conda install mamba -n base -c conda-forge (for more information, see: https://github.com/mamba-org/mamba)
3. Use the environment.yml file in this repository to create the MultiEyeTemAnalysis environment: mamba env create -f environment.yml

You can now run conda activate MultiEyeTemAnalysis and should have all required packages installed for running the analyses.

## Obtaining the Data

-- At the moment data is not publically available. The data is stored on the MPIB Share Server in the multieyetem folder smb://mpib-berlin.mpg.de/Share/Projects/879-multieyetem. Eyetracking data is saved in the data_eyelink folder and bheavioural data is saved in the folder called behavioural_outputfiles.

## Experiment code

## Order of running the scripts

1. 01_
2. 02_
3. 03_eyetracking_analysis?


# Explanation of Files

## General Files

- <.gitignore> --> which files to ignore during version control
- <.environment.yml> --> software dependencies, see "Installation"
- <README.md> --> general information about this repository

## Figure creation scripts

- <plots_fig1...>
