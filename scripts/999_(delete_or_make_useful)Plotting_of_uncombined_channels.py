# %%
# Importsconda
import mne
import numpy as np
import pandas as pd
from datetime import datetime
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt

# %% Load Metadata

df = pd.read_csv('../data/Tables_and_Dataframes/df.csv')
good_partis = df['subID'].unique()
new_partis = np.delete(good_partis, np.where(good_partis == ['KJN4WD']))

# %%
# if qt is used as backend the kernel crashes
# this can probably be deleted once run, but might just keep in in to be safe
mne.viz.set_browser_backend('matplotlib')

# %%


#  JUST FOR NOW - DELETed THE DIFFERENTLY SAMPLED EPOCH FROM THE FOLDER, copy folder still has it

# appending all epochs into one list and combining both pupil channels
con1_evoked_all_pp = []
con2_evoked_all_pp = []
con3_evoked_all_pp = []
con1_evoked_all_pp_list = []
con2_evoked_all_pp_list = []
con3_evoked_all_pp_list = []

all_con_evoked = pd.DataFrame()

for i in good_partis:
    
    # /mnt/home/bielefeldt/MPIB-SRT/879-multieyetem/multieyetem/data/Epochs_data_2024-03-20_14:25 -- is the new path after changing the epochs baseline to None, 
    # ran epochs with 011_Epoching
    # differently sampled subID epochs are removed in this folder too (but kept in the copy of that folder)
    folder_path = "/mnt/home/bielefeldt/MPIB-SRT/879-multieyetem/multieyetem/data/Epochs_pos1"

    epoch_file = folder_path + "/" + i + "_pos1-epo.fif"

    pupil_epoch = mne.read_epochs(epoch_file, preload=True, verbose=None)

    con1_epochs = pupil_epoch["items_presented == 1"]
    con2_epochs = pupil_epoch["items_presented == 2"]
    con3_epochs = pupil_epoch["items_presented == 3"]

    con1_ave = con1_epochs.average()
    con2_ave = con2_epochs.average()
    con3_ave = con3_epochs.average()

    con1_evoked_all_pp.append(con1_ave)
    con2_evoked_all_pp.append(con2_ave)
    con3_evoked_all_pp.append(con3_ave)
    
    con1_ave_df = con1_ave.to_data_frame()
    con1_ave_df['Load'] = 1
    con1_ave_df['subID'] = i
    con2_ave_df = con2_ave.to_data_frame()
    con2_ave_df['Load'] = 2
    con2_ave_df['subID'] = i
    con3_ave_df = con3_ave.to_data_frame()
    con3_ave_df['Load'] = 3
    con3_ave_df['subID'] = i

    con1_evoked_all_pp_list.append(con1_ave_df)
    con2_evoked_all_pp_list.append(con2_ave_df)
    con3_evoked_all_pp_list.append(con3_ave_df)


    all_con_evoked = pd.concat([all_con_evoked, con1_ave_df])
    all_con_evoked = pd.concat([all_con_evoked, con2_ave_df])
    all_con_evoked = pd.concat([all_con_evoked, con3_ave_df])

all_pp_con1 = all_con_evoked.loc[all_con_evoked['Load']==1]
all_pp_con2 = all_con_evoked.loc[all_con_evoked['Load']==2]
all_pp_con3 = all_con_evoked.loc[all_con_evoked['Load']==3]


evoked_dict = {1: [con1_evoked_all_pp_list], 2: [con2_evoked_all_pp_list], 3: [con3_evoked_all_pp_list]}


# %% 

# since changing the errorbar to 'se' it only takes a minute to plot

fig, ax = plt.subplots(figsize=(10,8))

# stefan included get_axes in his code to show axvlines, seems to work without it though and even returns an error 
# ax = fig.get_axes()[0]

sns.lineplot(data=all_con_evoked, x='time', y='pupil_right', errorbar='se')
sns.lineplot(data=all_con_evoked, x='time', y='pupil_left', errorbar='se')

ax.axvline(0, color='black', alpha=0.33)
ax.axvline(1, color='black', alpha=0.66)
ax.axvline(2, color='black')
ax.axvline(3, color='darkblue')
ax.axvline(6.5, color='blue')


# at the moment the text are in the middle of the plot and are overlapping each other.
# could maybe have them be printed at a diagonal on top of the plot?
label_y = 1.01
# Add labels above the vertical lines
ax.text(0, ax.get_ylim()[1] * label_y, "object position 1", color='black', alpha=0.33, fontsize=10, ha='center', rotation=45)
ax.text(1, ax.get_ylim()[1] * label_y, "object position 2", color='black', alpha=0.66, fontsize=10, ha='center', rotation=45)
ax.text(2, ax.get_ylim()[1] * label_y, "object position 3", color='black', fontsize=10, ha='center', rotation=45)
ax.text(3, ax.get_ylim()[1] * label_y, "onset delay", color='darkblue', fontsize=10, ha='center', rotation=45)
ax.text(6.5, ax.get_ylim()[1] * label_y, "onset probe object", color='blue', fontsize=10, ha='center', rotation=45)


name = 'pupil_fig_nocombine_channel.png'
figures_folder = '/mnt/home/bielefeldt/MPIB-SRT/879-multieyetem/multieyetem/Figures'
figure_filepath = os.path.join(figures_folder, name)
plt.suptitle('Pupil dilation uncombined channels', y=1.1, fontsize = 20)
plt.savefig(figure_filepath, bbox_inches = 'tight')

plt.show


# %%

# statistics on the dataframes?
# redo this as a repeated measures anova (check recording) and also potentially choose a certain time window instead of using all data


# anova_result = pg.anova(data=all_con_ave_df, dv='pupils', between='Load')
# anova_result

# %%
# Trial Df to test stuff for plot above - delete when above plot is finished

trial_df = pd.DataFrame({'time': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'col1': [ 1, 1, 1, 2, 2, 2, 3, 3, 3], 'col2': [13, 14, 22, 26, 30, 12, 11, 9, 33]})


fig, ax = plt.subplots(figsize=(10,8))

palette = sns.color_palette('husl', 3)
sns.lineplot(data=trial_df, x='time', y='col2', hue='col1', palette=palette, errorbar='se')

ax.axvline(0, label="onset probe object", color='darkblue')
ax.axvline(3.5, label="onset delay", color='green')
ax.axvline(4.5, color='black')  # Customize color and saturation
ax.axvline(5.5, color='black', alpha=0.66)  # Customize color and saturation
ax.axvline(6.5, color='black', alpha=0.33)  # Customize color and saturation

ax.text(4.5, 10, "object position 3", color='black', fontsize=10, ha='center',  rotation=45)
plt.savefig('trial_df_fig.png', bbox_inches = 'tight')


# %%
#

# %% 

# Not working - using mne.viz.plot
# combined plot showing all conditions overlapping

# evokeds_list = [con1_ave, con2_ave, con3_ave]

# the following is a list of a list of evokeds (each con1_evoked_all_pp is one list of evokeds):
# evokeds_list = [con1_evoked_all_pp, con2_evoked_all_pp, con3_evoked_all_pp]

# conds = ("1", "2", "3")

# evks = dict(zip(conds, evokeds_list))

# mne.viz.plot_compare_evokeds(evks)
