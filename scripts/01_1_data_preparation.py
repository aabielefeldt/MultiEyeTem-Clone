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

# Changing content of accuracy column where it is 2 to nan
# Note: Good idea??
df['acc_trial_concrete_cued'].replace(2, np.nan, inplace=True)

# %% 
# Adding Columns for later use

# %%
# Adding Column for mask
df['mask'] = 0
df.loc[df['onset_mask'].notna() & (df['onset_mask'] > 0), 'mask'] = 1


# %% 
# Adding Column for Button Press
# could have also used the column "trigger_object_concrete" which has as its content 220 and 221 - the triggers indicating wether the probe object was tilted to the right or to the left from the cued ori
df['correct_rot_response'] = df['object_cued_rot'] - df['object_test_rot_cued']

# Adding the new column with 0 and 1 values based on conditions
df['button_press_clockwise'] = 0  # Initialize with 0
# if the correct response is clockwise and the person was correct -> clockwise button press
df.loc[(df['correct_rot_response'] > 0) & (df['acc_trial_concrete_cued'] == 1), 'button_press_clockwise'] = 1
# if the correct response is anticlockwise and the person was not correct -> clockwise button press
df.loc[(df['correct_rot_response'] < 0) & (df['acc_trial_concrete_cued'] == 0), 'button_press_clockwise'] = 1
# dealing with nan
df.loc[pd.isna(df['acc_trial_concrete_cued']), 'button_press_clockwise'] = np.nan


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
# Adding column for orientation category of probe item

orientation_subset_cardinal = [348.75, 11.25, 78.75, 101.25, 168.75, 191.25, 258.75, 281.25]

# Create a new column with bin labels
df['probe_ori_cat'] = df['object_cued_rot'].apply(lambda x: np.nan if pd.isna(x) else 'cardinal' if x in orientation_subset_cardinal else 'oblique')


# %%
# Creating a column for cardinal vs oblique orientation category per shown item (encoding)

# orientations closest to the cardinal axes
orientation_subset_cardinal = [348.75, 11.25, 78.75, 101.25, 168.75, 191.25, 258.75, 281.25]

# Create a new column to indicate the categories
df['obj1_ori_cat'] = df['object_1_rot'].apply(lambda x: np.nan if pd.isna(x) else 'cardinal' if x in orientation_subset_cardinal else 'oblique')
df['obj2_ori_cat'] = df['object_2_rot'].apply(lambda x: np.nan if pd.isna(x) else 'cardinal' if x in orientation_subset_cardinal else 'oblique')
df['obj3_ori_cat'] = df['object_3_rot'].apply(lambda x: np.nan if pd.isna(x) else 'cardinal' if x in orientation_subset_cardinal else 'oblique')

# %% Add column for defining trial based on how many stimuli were either cardinal or oblique
# n for nans, c and o for cardinal and oblique (eg. in load 2: 'con')

df['trial_ori_cat_combo'] = df.apply(
    lambda row: ''.join(
        val[0] if isinstance(val, str) else 'n'
        for val in [row['obj1_ori_cat'], row['obj2_ori_cat'], row['obj3_ori_cat']]
    ),
    axis=1
)

#%%
# adding column for all_cardinal, all_oblique or mixed

df['trial_ori_pattern'] = np.where(
    df['trial_ori_cat_combo'].isin(['cnn', 'ccn', 'ccc']),
    'all_cardinal',
    np.where(
        df['trial_ori_cat_combo'].isin(['onn', 'oon', 'ooo']),
        'all_oblique',
        'mixed'
    )
)

#%%
mostly_cardinal = ['cnn', 'ccn', 'ccc', 'cco', 'coc', 'occ']
mostly_oblique = ['onn', 'oon', 'ooo', 'coo', 'oco', 'ooc']
category_map = {}
category_map.update({cat: 'mostly_cardinal' for cat in mostly_cardinal})
category_map.update({cat: 'mostly_oblique' for cat in mostly_oblique})

df['cardinal_type'] = df['trial_ori_cat_combo'].map(lambda x: category_map.get(x, 'mixed'))

#%%
# creating categorical columns depending on probes and sample objects distance to the next cardinal and wether response
# pressed the objects further or closer towards cardinal 

CARDINALS = np.array([0.0, 90.0, 180.0, 270.0])

def wrap360(x):
    return np.mod(x, 360.0)

def circ_dist_deg(a, b):
    """Circular (unsigned) distance in degrees in [0, 180]. Works on arrays."""
    a = wrap360(a)
    b = wrap360(b)
    d = np.abs(a - b)
    return np.minimum(d, 360.0 - d)

def nearest_cardinal_axis(theta):
    """Nearest cardinal axis (0/90/180/270) for each theta (array)."""
    theta = wrap360(theta)
    dists = np.stack([circ_dist_deg(theta, c) for c in CARDINALS], axis=1)
    idx = np.argmin(dists, axis=1)
    return CARDINALS[idx]

# --- vectors ---
mem   = df["object_cued_rot"].to_numpy(dtype=float)          # remembered orientation
probe = df["object_test_rot_cued"].to_numpy(dtype=float)     # probe shown at test
resp  = df["final_rot_concrete_cued"].to_numpy(dtype=float)  # final response

# axis defined by the remembered orientation (your request)
axis = nearest_cardinal_axis(mem)

# distance of probe and response from that axis
d_mem   = circ_dist_deg(mem, axis)
d_probe = circ_dist_deg(probe, axis)
resp_weighted = (probe+((probe-resp)/8)) # this treats the button press from the probe orientation as -1 or +1 degree to
# avoid that probe objects close to cardinals are treated as further away from cardinal axis (this makes the movement more 
# interested in directions towards or away from closest cardinal rather than actual distance of response to probe)
# object cued = 11.25, probe = 3.25, after button press towards cardinal orientation of response would be 4.75,
# i treat it as 2.25 (so moved towards cardinal by one degree instead of 8)
d_resp  = circ_dist_deg(resp_weighted, axis)

# add columns
df["nearest_cardinal_mem"] = axis
df["dist_to_cardinal_mem"] = d_mem
df['dist_probe_to_cardinal'] = d_probe

# 1 = probe is farther from cardinal (moved away)
# 0 = probe is closer to cardinal (moved toward)
probe_away = (d_probe > d_mem).astype(float)
# Handle rare ties (should not occur with ±8° offsets, but safe)
probe_away[np.isclose(d_probe, d_mem, atol=1e-12)] = np.nan
df["probe_away_from_cardinal"] = probe_away

# Away/toward based on the *movement from probe to response*
# 1 = moved away from cardinal, 0 = moved toward cardinal
pressed_away = (d_probe > d_resp).astype(float)
pressed_away[np.isnan(resp)] = np.nan  # keep missing responses as NaN
# Optional: if someone ended exactly same distance (rare), mark NaN
pressed_away[np.isclose(d_resp, d_probe, atol=1e-12)] = np.nan
df["pressed_away_from_cardinal"] = pressed_away

# %% 
# Creating folder structure and saving the dataframe

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

