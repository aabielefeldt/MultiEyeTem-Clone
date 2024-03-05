# %%
# Importsconda
import mne
import numpy as np
import pandas as pd
from datetime import datetime

# %% Load Metadata

df = pd.read_csv('../data/df.csv')
good_partis = df['subID'].unique()

# %%
# if qt is used as backend the kernel crashes
# this can probably be deleted once run, but might just keep in in to be safe
mne.viz.set_browser_backend('matplotlib')



# %% Custom Mappings

# this might not be all of the triggers
custom_mapping = {
    '1': 1, '10': 10, '100': 100, '101': 101, '102': 102, '103': 103, '104': 104, '105': 105, '106': 106, '107': 107,
    '108': 108, '109': 109, '11': 11, '110': 110, '111': 111, '112': 112, '113': 113, '114': 114, '115': 115,
    '116': 116, '117': 117, '118': 118, '119': 119, '12': 12, '120': 120, '121': 121, '122': 122, '123': 123,
    '124': 124, '125': 125, '126': 126, '127': 127, '128': 128, '129': 129, '13': 13, '130': 130, '131': 131,
    '132': 132, '133': 133, '134': 134, '135': 135, '136': 136, '137': 137, '138': 138, '139': 139, '14': 14,
    '140': 140, '141': 141, '142': 142, '143': 143, '144': 144, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19,
    '2': 2, '20': 20, '21': 21, '22': 22, '220': 220, '221': 221, '23': 23, '24': 24, '241': 241, '242': 242,
    '244': 244, '245': 245, '246': 246, '25': 25, '26': 26, '27': 27, '28': 28, '29': 29, '3': 3, '30': 30,
    '31': 31, '32': 32, '33': 33, '34': 34, '35': 35, '36': 36, '37': 37, '38': 38, '39': 39, '4': 4, '40': 40,
    '41': 41, '42': 42, '43': 43, '44': 44, '45': 45, '46': 46, '47': 47, '48': 48, '49': 49, '5': 5, '50': 50,
    '51': 51, '52': 52, '53': 53, '54': 54, '55': 55, '56': 56, '57': 57, '58': 58, '59': 59, '6': 6, '60': 60,
    '61': 61, '62': 62, '63': 63, '64': 64, '65': 65, '66': 66, '67': 67, '68': 68, '69': 69, '7': 7, '70': 70,
    '71': 71, '72': 72, '73': 73, '74': 74, '75': 75, '76': 76, '77': 77, '78': 78, '79': 79, '8': 8, '80': 80,
    '81': 81, '82': 82, '83': 83, '84': 84, '85': 85, '86': 86, '87': 87, '88': 88, '89': 89, '9': 9, '90': 90,
    '91': 91, '92': 92, '93': 93, '94': 94, '95': 95, '96': 96, '97': 97, '98': 98, '99': 99, 'fixation': 152,
    'saccade': 153,
    '220': 220, "221": 221, '241': 241, '242': 242, '244': 244, '245': 245, '246': 246
}


# %% 
# Dictionary for triggers used for onset of epoch?
# could also be a dict however:
test_onset_dict = {"onset_concrete_cued_left": 220,
    "onset_concrete_cued_right": 221}

# %%

# Get current date
current_date = datetime.now().strftime("%Y-%m-%d_%H:%M")

# Create folder name using current date
folder_name = f"Epochs_data_{current_date}"

# Create the folder
os.makedirs(folder_name)

# Looping over all subIDs
# use once I have my epochs correct
freqs = []
# for i in good_partis:
one_subID = ['REUNMA']
for i in good_partis:
    folder_path = "/mnt/home/bielefeldt/MPIB-SRT/879-multieyetem/data_eyelink"

    eyelink_file = folder_path + "/" + i + ".asc"
    print(eyelink_file)
    raw = mne.io.read_raw_eyelink(eyelink_file)

    metadata_pp = df.loc[(df['subID'] == i)]
    freq_pp = [i, (raw.info['sfreq'])]
    freqs.append(freq_pp)
    events_from_annot, event_dict = mne.events_from_annotations(
    raw, event_id=custom_mapping)

    # need to somehow save this with reference to the subID
    pupil_epochs = mne.Epochs(
        raw,
        # had to collapse the events from annot to the length of the events of interest (test_onset_dict), seems to be a bug in mne (report?)
        events_from_annot[np.isin(events_from_annot, [220, 221])[:,-1]],
        event_id=test_onset_dict,
        tmin=-6.5,
        tmax=2.5,

        metadata=metadata_pp,
        picks=['pupil_left', 'pupil_right'],
        event_repeated='merge',
        preload=True,
        reject_by_annotation=False
    )
    # save epoch in fif format to a specified folder
    
    pupil_epochs.plot(n_epochs=4)

    # save 
    filename = f"{i}-epo.fif"
    filepath = os.path.join(folder_name, filename)
    pupil_epochs.save(filepath, overwrite=True, split_size='1.8GB', split_naming='bids', verbose=None)
freqs_df = pd.DataFrame(freqs, columns=['subID', 'frequency'])
display(freqs_df)
# only one has a different sampling freuqency and that one is 1000hz so higher than the rest (500hz)´

#%%

def count_files_in_folder(folder_path):
    # Initialize count
    file_count = 0
    
    # Iterate through the files in the folder
    for _, _, files in os.walk(folder_path):
        file_count += len(files)
    
    return file_count

# Example usage:
# folder_path = 'path/to/your/folder'
num_files = count_files_in_folder(folder_name)
print(f"Number of files in the folder: {num_files}")

# %%
# trying to average the data per participant into one epoch?
# is that baselining? Or is that something else

# trying baselining
## "To reduce the effect of such drifts, it is custom to perform a so-called baseline correction. 
# Essentially, this consists of using EEG activity over a baseline period, i.e. before an external 
# event occurs, to correct activity over a post-stimulus interval, i.e. the time after an external 
# event occurs."

folder_name = 'Epochs_data_2024-02-29_13:54'
file_name = 'REUNMA-epo.fif'

filepath = os.path.join(folder_name, file_name)

# %%

filepath = "/mnt/home/bielefeldt/MPIB-SRT/879-multieyetem/multieyetem/data/Epochs_data_2024-02-29_13:54/REUNMA-epo.fif"

pupil_epochs = mne.read_epochs(filepath, proj=True, preload=True, verbose=None)

baseline_tmin, baseline_tmax = None, 0  # None takes the first timepoint

baseline = (baseline_tmin, baseline_tmax)

trad_aud = pupil_epochs.average().apply_baseline(baseline)

# %%
# this did some sort of average over all 576 trials - but seperately for the two pupils and for all conditions together
# need to figure out what the baseline means and also seperate by condition first

trad_aud.plot()

# %%
evoked = pupil_epochs.average() # original line of code: epochs["auditory/left"].average() - ["auditory/left"] --> whats that in the square brackets, 
# how can I use that in my code
# why is this plotted twice? The two lines should probably already be the two individual eyes
evoked.plot()

# this works:
# evoked = pupil_epochs['onset_concrete_cued_right'].average()
# evoked


# lookd exactly like the baseline???

# %%
# trying to combine the two eyes into one - plotting should show only one line
# the following does not work
pupil_epochs.combine_event_ids(pupil_epochs, ['220', '221'], {'onset': 12})


# %%

# why does this not work
groups=dict(Pupils=['pupil_left', 'pupil_right'])

mne.channels.combine_channels(pupil_epochs, groups, method='mean') #, keep_stim=False, drop_bad=False, verbose=None)



# %% 
# this works for combining the two channels into one
# This works... why though?

pupil_ix = mne.pick_channels(pupil_epochs.info["ch_names"], include=['pupil_left', 'pupil_right'])

roi_dict = dict(left_ROI=pupil_ix)
roi_evoked = mne.channels.combine_channels(pupil_epochs, roi_dict, method="mean")
roi_evoked.plot()
ave = roi_evoked.average()
ave.plot()

# %%
ave_df = ave.to_data_frame()# %%
# This transforms the epoch data to a dataframe to then potentially use for a seaborn function
# this however is at the moment all 576 epochs one after the other and still both eyes
# this could probably be used as is though if I would just calculate the mean for both eyes and put it into a new column
# and also calculate the mean over all epochs
# this might lose some of the usability of mne though and the epochs_df at the moment is very large
epochs_df = pupil_epochs.to_data_frame()

# %% 
# looks exactly the same
# I first tried to do this with pupil_left and pupil_right

l_pup = pupil_epochs['onset_concrete_cued_left', 'onset_concrete_cued_right'].average()
l_pup.plot()





#%%

# %%

# Get current date
current_date = datetime.now().strftime("%Y-%m-%d_%H:%M")

# Create folder name using current date
folder_name = f"Epochs_data_{current_date}"

# Create the folder
os.makedirs(folder_name)

# Looping over all subIDs
# use once I have my epochs correct
freqs = []
# for i in good_partis:
one_subID = ['REUNMA']
for i in one_subID:
    folder_path = "/mnt/home/bielefeldt/MPIB-SRT/879-multieyetem/data_eyelink"

    eyelink_file = folder_path + "/" + i + ".asc"
    print(eyelink_file)
    raw = mne.io.read_raw_eyelink(eyelink_file)

    metadata_pp = df.loc[(df['subID'] == i)]
    freq_pp = [i, (raw.info['sfreq'])]
    freqs.append(freq_pp)
    events_from_annot, event_dict = mne.events_from_annotations(
    raw, event_id=custom_mapping)

    # need to somehow save this with reference to the subID
    pupil_epochs_all_ids = mne.Epochs(
        raw,
        # had to collapse the events from annot to the length of the events of interest (test_onset_dict), seems to be a bug in mne (report?)
        events_from_annot,
        tmin=-6.5,
        tmax=2.5,
        picks=['pupil_left', 'pupil_right', 'DIN'],
        event_repeated='merge',
        preload=True,
        reject_by_annotation=False
    )

    # 

    # save epoch in fif format to a specified folder
    
    pupil_epochs.plot(n_epochs=4)

    # save 
    filename = f"{i}-epo.fif"
    filepath = os.path.join(folder_name, filename)
    pupil_epochs.save(filepath, overwrite=True, split_size='1.8GB', split_naming='bids', verbose=None)
freqs_df = pd.DataFrame(freqs, columns=['subID', 'frequency'])
display(freqs_df)











# %%
# Plot
# with the top line it plots the data without annotations - might be able to also use this to only plot the data with triggers but without the other annotations (fix, sacc, and blinks)
# however the following line makes events_from_annotation not work ?!

# raw.set_annotations(None)  # without this, it's really slow
# raw.plot()

picks_idx = mne.pick_channels(
    raw.ch_names, ['pupil_left', 'pupil_right'], ordered=True
)

raw.plot(events= events_from_annot, event_id=event_dict, order=picks_idx)
# %%
# events from annotations without any custom mapping dictionaries
# mne.events reads both the events and the events_dict in at the same time as a tuple ([events_from_annot], {event_dict})
# -> I am simply unpacking them already into two seperate variables
events_from_annot, event_dict = mne.events_from_annotations(raw, event_id='auto')
print(event_dict)
print(events_from_annot[:5])

#
# %%
# # mapping the triggers (in quotations marks) to the same numbers as event ids to make it more useful
# this might not be all of the triggers
custom_mapping = {
    '1': 1, '10': 10, '100': 100, '101': 101, '102': 102, '103': 103, '104': 104, '105': 105, '106': 106, '107': 107,
    '108': 108, '109': 109, '11': 11, '110': 110, '111': 111, '112': 112, '113': 113, '114': 114, '115': 115,
    '116': 116, '117': 117, '118': 118, '119': 119, '12': 12, '120': 120, '121': 121, '122': 122, '123': 123,
    '124': 124, '125': 125, '126': 126, '127': 127, '128': 128, '129': 129, '13': 13, '130': 130, '131': 131,
    '132': 132, '133': 133, '134': 134, '135': 135, '136': 136, '137': 137, '138': 138, '139': 139, '14': 14,
    '140': 140, '141': 141, '142': 142, '143': 143, '144': 144, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19,
    '2': 2, '20': 20, '21': 21, '22': 22, '220': 220, '221': 221, '23': 23, '24': 24, '241': 241, '242': 242,
    '244': 244, '245': 245, '246': 246, '25': 25, '26': 26, '27': 27, '28': 28, '29': 29, '3': 3, '30': 30,
    '31': 31, '32': 32, '33': 33, '34': 34, '35': 35, '36': 36, '37': 37, '38': 38, '39': 39, '4': 4, '40': 40,
    '41': 41, '42': 42, '43': 43, '44': 44, '45': 45, '46': 46, '47': 47, '48': 48, '49': 49, '5': 5, '50': 50,
    '51': 51, '52': 52, '53': 53, '54': 54, '55': 55, '56': 56, '57': 57, '58': 58, '59': 59, '6': 6, '60': 60,
    '61': 61, '62': 62, '63': 63, '64': 64, '65': 65, '66': 66, '67': 67, '68': 68, '69': 69, '7': 7, '70': 70,
    '71': 71, '72': 72, '73': 73, '74': 74, '75': 75, '76': 76, '77': 77, '78': 78, '79': 79, '8': 8, '80': 80,
    '81': 81, '82': 82, '83': 83, '84': 84, '85': 85, '86': 86, '87': 87, '88': 88, '89': 89, '9': 9, '90': 90,
    '91': 91, '92': 92, '93': 93, '94': 94, '95': 95, '96': 96, '97': 97, '98': 98, '99': 99, 'fixation': 152,
    'saccade': 153,
    '220': 220, "221": 221, '241': 241, '242': 242, '244': 244, '245': 245, '246': 246
}

# %%

events_from_annot, event_dict = mne.events_from_annotations(
    raw, event_id=custom_mapping
)
print(event_dict)
print(events_from_annot[:5])

for annot in raw.annotations:
    print(annot["onset"], annot["duration"], annot["description"])

# %%
# using the same strategy again to now map the events where the event ids are the same as the trigger numbers to the strings
# I dont think I can pass the events to the events_from_annotations because for that I need to pass the raw directly to the function


# %%
# I can then transform the events back to an annotations object and there I can pass the actual descriptive strings to the now correct trigger numbers (event ids).
# Am I correct in believing that the event ID's are not the triggers but are what I called them before in the dictionary above?. Not sure is that is useful
# These return annotations
# Am I using them afterwards at all?
custom_event_mapping = {
    1: "obj1/pos1/ori1", 2: "obj1/pos1/ori2", 3: "obj1/pos1/ori3", 4: "obj1/pos1/ori4",
    5: "obj1/pos1/ori5", 6: "obj1/pos1/ori6", 7: "obj1/pos1/ori7", 8: "obj1/pos1/ori8",
    9: "obj1/pos1/ori9", 10: "obj1/pos1/ori10", 11: "obj1/pos1/ori11", 12: "obj1/pos1/ori12",
    13: "obj1/pos1/ori13", 14: "obj1/pos1/ori14", 15: "obj1/pos1/ori15", 16: "obj1/pos1/ori16",
    17: "obj2/pos1/ori1", 18: "obj2/pos1/ori2", 19: "obj2/pos1/ori3", 20: "obj2/pos1/ori4",
    21: "obj2/pos1/ori5", 22: "obj2/pos1/ori6", 23: "obj2/pos1/ori7", 24: "obj2/pos1/ori8",
    25: "obj2/pos1/ori9", 26: "obj2/pos1/ori10", 27: "obj2/pos1/ori11", 28: "obj2/pos1/ori12",
    29: "obj2/pos1/ori13", 30: "obj2/pos1/ori14", 31: "obj2/pos1/ori15", 32: "obj2/pos1/ori16",
    33: "obj3/pos1/ori1", 34: "obj3/pos1/ori2", 35: "obj3/pos1/ori3", 36: "obj3/pos1/ori4",
    37: "obj3/pos1/ori5", 38: "obj3/pos1/ori6", 39: "obj3/pos1/ori7", 40: "obj3/pos1/ori8",
    41: "obj3/pos1/ori9", 42: "obj3/pos1/ori10", 43: "obj3/pos1/ori11", 44: "obj3/pos1/ori12",
    45: "obj3/pos1/ori13", 46: "obj3/pos1/ori14", 47: "obj3/pos1/ori15", 48: "obj3/pos1/ori16",
    49: "obj1/pos2/ori1", 50: "obj1/pos2/ori2", 51: "obj1/pos2/ori3", 52: "obj1/pos2/ori4",
    53: "obj1/pos2/ori5", 54: "obj1/pos2/ori6", 55: "obj1/pos2/ori7", 56: "obj1/pos2/ori8",
    57: "obj1/pos2/ori9", 58: "obj1/pos2/ori10", 59: "obj1/pos2/ori11", 60: "obj1/pos2/ori12",
    61: "obj1/pos2/ori13", 62: "obj1/pos2/ori14", 63: "obj1/pos2/ori15", 64: "obj1/pos2/ori16",
    65: "obj2/pos2/ori1", 66: "obj2/pos2/ori2", 67: "obj2/pos2/ori3", 68: "obj2/pos2/ori4",
    69: "obj2/pos2/ori5", 70: "obj2/pos2/ori6", 71: "obj2/pos2/ori7", 72: "obj2/pos2/ori8",
    73: "obj2/pos2/ori9", 74: "obj2/pos2/ori10", 75: "obj2/pos2/ori11", 76: "obj2/pos2/ori12",
    77: "obj2/pos2/ori13", 78: "obj2/pos2/ori14", 79: "obj2/pos2/ori15", 80: "obj2/pos2/ori16",
    81: "obj3/pos2/ori1", 82: "obj3/pos2/ori2", 83: "obj3/pos2/ori3", 84: "obj3/pos2/ori4",
    85: "obj3/pos2/ori5", 86: "obj3/pos2/ori6", 87: "obj3/pos2/ori7", 88: "obj3/pos2/ori8",
    89: "obj3/pos2/ori9", 90: "obj3/pos2/ori10", 91: "obj3/pos2/ori11", 92: "obj3/pos2/ori12",
    93: "obj3/pos2/ori13", 94: "obj3/pos2/ori14", 95: "obj3/pos2/ori15", 96: "obj3/pos2/ori16",
    97: "obj1/pos3/ori1", 98: "obj1/pos3/ori2", 99: "obj1/pos3/ori3", 100: "obj1/pos3/ori4",
    101: "obj1/pos3/ori5", 102: "obj1/pos3/ori6", 103: "obj1/pos3/ori7", 104: "obj1/pos3/ori8",
    105: "obj1/pos3/ori9", 106: "obj1/pos3/ori10", 107: "obj1/pos3/ori11", 108: "obj1/pos3/ori12",
    109: "obj1/pos3/ori13", 110: "obj1/pos3/ori14", 111: "obj1/pos3/ori15", 112: "obj1/pos3/ori16",
    113: "obj2/pos3/ori1", 114: "obj2/pos3/ori2", 115: "obj2/pos3/ori3", 116: "obj2/pos3/ori4",
    117: "obj2/pos3/ori5", 118: "obj2/pos3/ori6", 119: "obj2/pos3/ori7", 120: "obj2/pos3/ori8",
    121: "obj2/pos3/ori9", 122: "obj2/pos3/ori10", 123: "obj2/pos3/ori11", 124: "obj2/pos3/ori12",
    125: "obj2/pos3/ori13", 126: "obj2/pos3/ori14", 127: "obj2/pos3/ori15", 128: "obj2/pos3/ori16",
    129: "obj3/pos3/ori1", 130: "obj3/pos3/ori2", 131: "obj3/pos3/ori3", 132: "obj3/pos3/ori4",
    133: "obj3/pos3/ori5", 134: "obj3/pos3/ori6", 135: "obj3/pos3/ori7", 136: "obj3/pos3/ori8",
    137: "obj3/pos3/ori9", 138: "obj3/pos3/ori10", 139: "obj3/pos3/ori11", 140: "obj3/pos3/ori12",
    141: "obj3/pos3/ori13", 142: "obj3/pos3/ori14", 143: "obj3/pos3/ori15", 144: "obj3/pos3/ori16",

    220: "onset_concrete_cued_left", 221: "onset_concrete_cued_right", 241: "left_button_press",
    242: "right_button_press", 244: "feedback_cued_item", 245: "mask", 246: "beginning/end_of_task"
}

annot_from_events = mne.annotations_from_events(events_from_annot, 500, event_desc=custom_event_mapping)

for annot in annot_from_events:
    print(annot["onset"], annot["duration"], annot["description"])


# %%
custom_scalings = dict(eyegaze=1e3)
raw.pick(picks="eyetrack").plot(scalings=custom_scalings)

# %%

# maybe I can create a new raw object based on the new annotations and the new events?


# %%
# Epoching the data

# epochs = mne.Epochs(
#     raw,
#     events,
#     event_id=event_dict,
#     tmin=-0.2,
#     tmax=0.5,
#     reject=reject_criteria,
#     preload=True,
# )
# but here we’ll force loading into memory using the preload=True parameter so that we can see the results of the rejection criteria being applied:


# how can I epoch the data according to a specific trigger? This looks like it would just epoch around every single trigger which
# would then even include the fixations??
# I can then plot only the data for epochs I am interested in (meaning according to the triggers)
# still seems unnnecessary to epoch everything at once since the length of the epoch is only fitting for the epoch from stim 1 to test 1
epochs = mne.Epochs(
    raw,
    events_from_annot,
    event_id={'test_onset_1': 221, 'test_onset_2': 220},
    tmin=-5.5,
    tmax=1.5,
    picks=['pupil_left', 'pupil_right'],
    event_repeated='merge',
    metadata=df,
    preload=True,
    reject_by_annotation=False
)

#check what merge does exactly

#RuntimeError: Event time samples were not unique. Consider setting the `event_repeated` parameter."
# event_repeatedstr
# How to handle duplicates in events[:, 0]. Can be 'error' (default), to raise an error, ‘drop’ to only retain the row occurring first in the events, or 'merge' to combine the coinciding events (=duplicates) into a new event (see Notes for details).

# %%

events_from_annot
# %%
for event in events_from_annot:
    print(event)

#'fixation': 152

# %%
# this plots the triggers only, meg=False is weird since this is not MEG data to begin with
# however this finds the triggers as the stim channel straight away, no need to get them from the annotations
# what does raw.copy do?
# also: NOTE: pick_types() is a legacy function. New code should use inst.pick(...).
# however the raw.copy example by MNE returns a plot with the individual magnitudes of the stims, not as is done here as coloured lines

# raw.copy().pick_types(meg=False, stim=True).plot(start=3, duration=6)

# this still gives me an empty array even though the line above (raw.copy...) gives me the DIN channel with the triggers
# apparently it is normal for some systems to store the triggers as annotations so I won't need the following but will need events_from_annotations instead
# events = mne.find_events(raw, stim_channel="DIN")
# print(events[:5])  # show the first 5

# %%
print(events_from_annot[:, 0])




# %%
# dictionary to be used when epoching data?
ttl_dict = {
    "obj1/pos1/ori1": 1, "obj1/pos1/ori2": 2, "obj1/pos1/ori3": 3, "obj1/pos1/ori4": 4,
    "obj1/pos1/ori5": 5, "obj1/pos1/ori6": 6, "obj1/pos1/ori7": 7, "obj1/pos1/ori8": 8,
    "obj1/pos1/ori9": 9, "obj1/pos1/ori10": 10, "obj1/pos1/ori11": 11, "obj1/pos1/ori12": 12,
    "obj1/pos1/ori13": 13, "obj1/pos1/ori14": 14, "obj1/pos1/ori15": 15, "obj1/pos1/ori16": 16,
    "obj2/pos1/ori1": 17, "obj2/pos1/ori2": 18, "obj2/pos1/ori3": 19, "obj2/pos1/ori4": 20,
    "obj2/pos1/ori5": 21, "obj2/pos1/ori6": 22, "obj2/pos1/ori7": 23, "obj2/pos1/ori8": 24,
    "obj2/pos1/ori9": 25, "obj2/pos1/ori10": 26, "obj2/pos1/ori11": 27, "obj2/pos1/ori12": 28,
    "obj2/pos1/ori13": 29, "obj2/pos1/ori14": 30, "obj2/pos1/ori15": 31, "obj2/pos1/ori16": 32,
    "obj3/pos1/ori1": 33, "obj3/pos1/ori2": 34, "obj3/pos1/ori3": 35, "obj3/pos1/ori4": 36,
    "obj3/pos1/ori5": 37, "obj3/pos1/ori6": 38, "obj3/pos1/ori7": 39, "obj3/pos1/ori8": 40,
    "obj3/pos1/ori9": 41, "obj3/pos1/ori10": 42, "obj3/pos1/ori11": 43, "obj3/pos1/ori12": 44,
    "obj3/pos1/ori13": 45, "obj3/pos1/ori14": 46, "obj3/pos1/ori15": 47, "obj3/pos1/ori16": 48,
    "obj1/pos2/ori1": 49, "obj1/pos2/ori2": 50, "obj1/pos2/ori3": 51, "obj1/pos2/ori4": 52,
    "obj1/pos2/ori5": 53, "obj1/pos2/ori6": 54, "obj1/pos2/ori7": 55, "obj1/pos2/ori8": 56,
    "obj1/pos2/ori9": 57, "obj1/pos2/ori10": 58, "obj1/pos2/ori11": 59, "obj1/pos2/ori12": 60,
    "obj1/pos2/ori13": 61, "obj1/pos2/ori14": 62, "obj1/pos2/ori15": 63, "obj1/pos2/ori16": 64,
    "obj2/pos2/ori1": 65, "obj2/pos2/ori2": 66, "obj2/pos2/ori3": 67, "obj2/pos2/ori4": 68,
    "obj2/pos2/ori5": 69, "obj2/pos2/ori6": 70, "obj2/pos2/ori7": 71, "obj2/pos2/ori8": 72,
    "obj2/pos2/ori9": 73, "obj2/pos2/ori10": 74, "obj2/pos2/ori11": 75, "obj2/pos2/ori12": 76,
    "obj2/pos2/ori13": 77, "obj2/pos2/ori14": 78, "obj2/pos2/ori15": 79, "obj2/pos2/ori16": 80,
    "obj3/pos2/ori1": 81, "obj3/pos2/ori2": 82, "obj3/pos2/ori3": 83, "obj3/pos2/ori4": 84,
    "obj3/pos2/ori5": 85, "obj3/pos2/ori6": 86, "obj3/pos2/ori7": 87, "obj3/pos2/ori8": 88,
    "obj3/pos2/ori9": 89, "obj3/pos2/ori10": 90, "obj3/pos2/ori11": 91, "obj3/pos2/ori12": 92,
    "obj3/pos2/ori13": 93, "obj3/pos2/ori14": 94, "obj3/pos2/ori15": 95, "obj3/pos2/ori16": 96,
    "obj1/pos3/ori1": 97, "obj1/pos3/ori2": 98, "obj1/pos3/ori3": 99, "obj1/pos3/ori4": 100,
    "obj1/pos3/ori5": 101, "obj1/pos3/ori6": 102, "obj1/pos3/ori7": 103, "obj1/pos3/ori8": 104,
    "obj1/pos3/ori9": 105, "obj1/pos3/ori10": 106, "obj1/pos3/ori11": 107, "obj1/pos3/ori12": 108,
    "obj1/pos3/ori13": 109, "obj1/pos3/ori14": 110, "obj1/pos3/ori15": 111, "obj1/pos3/ori16": 112,
    "obj2/pos3/ori1": 113, "obj2/pos3/ori2": 114, "obj2/pos3/ori3": 115, "obj2/pos3/ori4": 116,
    "obj2/pos3/ori5": 117, "obj2/pos3/ori6": 118, "obj2/pos3/ori7": 119, "obj2/pos3/ori8": 120,
    "obj2/pos3/ori9": 121, "obj2/pos3/ori10": 122, "obj2/pos3/ori11": 123, "obj2/pos3/ori12": 124,
    "obj2/pos3/ori13": 125, "obj2/pos3/ori14": 126, "obj2/pos3/ori15": 127, "obj2/pos3/ori16": 128,
    "obj3/pos3/ori1": 129, "obj3/pos3/ori2": 130, "obj3/pos3/ori3": 131, "obj3/pos3/ori4": 132,
    "obj3/pos3/ori5": 133, "obj3/pos3/ori6": 134, "obj3/pos3/ori7": 135, "obj3/pos3/ori8": 136,
    "obj3/pos3/ori9": 137, "obj3/pos3/ori10": 138, "obj3/pos3/ori11": 139, "obj3/pos3/ori12": 140,
    "obj3/pos3/ori13": 141, "obj3/pos3/ori14": 142, "obj3/pos3/ori15": 143, "obj3/pos3/ori16": 144,
    "onset_concrete_cued_left": 220,
    "onset_concrete_cued_right": 221, "left_button_press": 241, "right_button_press": 242, "feedback_cued_item": 244,
    "mask": 245, "beginning/end_of_task": 246
}

# %%

# %% trying to make event specific epochs (i think?)


# somehow I now wanna use the ttl_dict to epoch my data?

# what about the settings to the epochs (time -0.5 ...)

test_onset = mne.pick_events(events_from_annot, include=[220, 221])

test_onset_id = {"onset_concrete_cued_left": 220, "onset_concrete_cued_right": 221}
# need to find out what my equivalent for the face is
epochs["fixation"].plot(
    # events=test_onset_id,
    events=test_onset,
    #event_id=event_dict,
    event_id=event_dict
    # event_color=dict("99"="red", "92"="blue"),
)
# KeyError: 'Event name "152" could not be found. The following events are present in the data: 1, 10, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 11, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 12, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 13, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 14, 140, 141, 142, 143, 144, 15, 16, 17, 18, 19, 2, 20, 21, 22, 220, 221, 23, 24, 241, 242, 244, 245, 246, 25, 26, 27, 28, 29, 3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 4, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 5, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 6, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 7, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 8, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 9, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, fixation, saccade, fixation/saccade, 51/fixation, 221/saccade, 60/fixation, 244/saccade, 241/saccade, 221/fixation'

# why are there two plots? one for pupil_left and one for pupil_right maybe because I specified those in the epochs above? But they are both in both plots
# plots look identical


# %%

print(event_dict)

# %%


#interesting events to consider for pupillopetry
# this is stolen from above
test_onset = mne.pick_events(events_from_annot, include=[220, 221])

# could also be a dict however:
test_onset_dict = {"onset_concrete_cued_left": 220,
    "onset_concrete_cued_right": 221,}

#
pupil_epochs = mne.Epochs(
    raw,
    events_from_annot,
    event_id=test_onset_dict,
    tmin=-0.5,
    tmax=0.5,

    picks=['pupil_left', 'pupil_right'],
    event_repeated='merge',
    preload=True,
)

# reject by annotation


# With the code above:
# pupil_epochs["onset_concrete_cued_left"].plot returns
# KeyError: 'Event name "onset_concreete_cued_left" could not be found. (because typo)
# The following events are present in the data: onset_concrete_cued_left, onset_concrete_cued_right'
# %%
pupil_epochs["onset_concrete_cued_left"].plot()

# three plots??
#

# %%
pupil_epochs["onset_concrete_cued_left", "onset_concrete_cued_right"].plot()
# %%
pupil_epochs["onset_concrete_cued_left", "onset_concrete_cued_right"].plot()
# %%
