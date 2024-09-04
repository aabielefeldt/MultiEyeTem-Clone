# %% 
# # Preparations for Behavioural Analyses
# 
# 1. loading the data
# 2. Deciding which participants to exlude depending on binomial tests of their accuracy
# 3. Adding columns to the dataframe for later use
# 4. Descriptive Statistics before excluding participants
# 5. Excluding participants
# 

# %%
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats #(not in environment yet?)
import sys
from statsmodels.stats.multicomp import pairwise_tukeyhsd #(not in envirnoment yet)
import seaborn as sns
import pingouin as pg

# %%
behav_data_path = "/mnt/home/bielefeldt/MPIB-SRT/879-multieyetem/behavioural_outputfiles"

df_list = []
for file_name in os.listdir(behav_data_path):
    if file_name.endswith(".txt") and "pr.txt" not in file_name and not file_name.startswith("resultfile_pil"):  # Check if the file has .txt extension
        file_path = os.path.join(behav_data_path, file_name)

        # Read the file and determine whether to skip rows based on numerical content
        data = pd.read_table(file_path, skipinitialspace=True)
        has_numerical_content = data.map(lambda x: isinstance(x, (int, float))).any(axis=1)

        # Exclude rows with no numerical content
        data = data[has_numerical_content]

        df_list.append(data)

# Concatenate all DataFrames in the list
df = pd.concat(df_list, ignore_index=True)

# %% 
# Adding Columns for later use

# Adding Column for mask
df['mask'] = 0
df.loc[df['onset_mask'].notna() & (df['onset_mask'] > 0), 'mask'] = 1

# %% 
# Adding Column for Button Press
# Note: There are no NANs in the button press column. What happens when no button was pressed? Probably they just get an inaccurate but this would not be seen in the data, without the trigger there would be no way of knowing whether it was inaccurate because no button was pressed or even though a button was pressed
# Note: might be in the df already??
# Calculate 'correct_rot_response'
df['correct_rot_response'] = df['object_cued_rot'] - df['object_test_rot_cued']

# Adding the new column with 0 and 1 values based on conditions
df['button_press_clockwise'] = 0  # Initialize with 0
# if the correct response is clockwise and the person was correct -> clockwise button press
df.loc[(df['correct_rot_response'] > 0) & (df['acc_trial_concrete_cued'] > 0), 'button_press_clockwise'] = 1
# if the correct response is anticlockwise and the person was not correct -> clockwise button press
df.loc[(df['correct_rot_response'] < 0) & (df['acc_trial_concrete_cued'] == 0), 'button_press_clockwise'] = 1

# %% 
# Adding column for stimuli subset
# Define the stimuli sets
stimuli_set1 = ['01_pitcher02a_BOSS.jpg', '01_lighthouse.jpg', '01_table03.jpg']
stimuli_set2 = ['02_crown.jpg', '02_candelabra.jpg', '02_photoshopTelephone.jpg']
stimuli_set3 = ['03_gazebo.jpg', '03_radio03a.jpg', '03_lamppost01.jpg']

# Function to determine the stimuli set based on 'object_1_name'
def determine_stimuli_set(row):
    object_name = row['object_1_name']
    if object_name in stimuli_set1:
        return 'stimuli_set1'
    elif object_name in stimuli_set2:
        return 'stimuli_set2'
    elif object_name in stimuli_set3:
        return 'stimuli_set3'
    else:
        return 'Unknown'

# Use the apply() method to create the 'stimuli_set' column
df['stimuli_set'] = df.apply(determine_stimuli_set, axis=1)

# %% 
# Adding column for orientation bin

# Define the orientation bins and labels
orientation_bins = {
    11.25: 'Ori Bin A', 33.75: 'Ori Bin B', 56.25: 'Ori Bin C', 78.75: 'Ori Bin D',
    101.25: 'Ori Bin A', 123.75: 'Ori Bin B', 146.25: 'Ori Bin C', 168.75: 'Ori Bin D',
    191.25: 'Ori Bin A', 213.75: 'Ori Bin B', 236.25: 'Ori Bin C', 258.75: 'Ori Bin D',
    281.25: 'Ori Bin A', 303.75: 'Ori Bin B', 326.25: 'Ori Bin C', 348.75: 'Ori Bin D'
}

# Create a new column with bin labels
df['Orientation_Bin'] = df['object_cued_rot'].map(orientation_bins)

# %% 
# Creating folder structure and saving the dataframe

# %%
# create data folder
multieyetem_folder = "/mnt/home/bielefeldt/MPIB-SRT/879-multieyetem/multieyetem"
data_foldername = f'data'
csv_path = os.path.join(multieyetem_folder, data_foldername)
os.makedirs(csv_path, exist_ok=True)

# create figures folder
multieyetem_folder = "/mnt/home/bielefeldt/MPIB-SRT/879-multieyetem/multieyetem"
data_foldername = f'figures'
csv_path = os.path.join(multieyetem_folder, data_foldername)
os.makedirs(csv_path, exist_ok=True)

# creating a folder for all table and dataframes that are created in this project
data_folder = "/mnt/home/bielefeldt/MPIB-SRT/879-multieyetem/multieyetem/data"
csv_foldername = f'csv_files'
csv_path = os.path.join(data_folder, csv_foldername)
os.makedirs(csv_path, exist_ok=True)

# Saving dataframe
df.to_csv('../data/csv_files/df.csv', index=False)


