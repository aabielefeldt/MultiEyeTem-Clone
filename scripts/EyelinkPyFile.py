# %%
# Importsconda 
import mne
import numpy as np

#  %% 
# Plotting only works if backend is set to matplotlib instead of qt
mne.viz.set_browser_backend('matplotlib')

# %%
# Print platform info
mne.sys_info()

# %%
# Read eyelink data
# eyelink_file = "REUNMA_copy_changed.asc"  # this is the file you provided for me, I just renamed it
folder_path = "/mnt/home/bielefeldt/MPIB-SRT/879-multieyetem/data_eyelink"

eyelink_file = folder_path + "/REUNMA.asc"
raw = mne.io.read_raw_eyelink(eyelink_file)
raw.load_data()

np.unique(raw.annotations.description)

# %%
# Print all annotations for inspection
for annot in raw.annotations:
    print(annot["onset"], annot["duration"], annot["description"])


# %%
# Plot
# raw.set_annotations(None)  # without this, it's really slow
raw.plot()

# %%
# Get MNE-Python test data
eyelink_test_data = mne.datasets.eyelink.data_path()
eyelink_test_file = eyelink_test_data / "sub-01_task-plr_eyetrack.asc"

raw2 = mne.io.read_raw_eyelink(eyelink_test_file)
raw2.load_data()

np.unique(raw2.annotations.description)

# %%
# Plot the MNE-Python test data
raw2.plot()
# %%
