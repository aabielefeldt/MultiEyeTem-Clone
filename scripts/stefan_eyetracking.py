"""Analyze the eyetracking data."""

# %%
# Imports
import pathlib

# %%
# Read data

# For example:
#     - when accessing via VPN --> smb://mpib-berlin.mpg.de/share/Projects/879-multieyetem/data_eyelink
#     - when on the server --> /mnt/home/USERNAME/MPIB-SRT/879-multieyetem/data_eyelink
#         - replace USERNAME with your username
#         - run the `amdmount` script to mount the MPIB-SRT folder first
#     - If you copied a data file to a local location --> /home/stefanappelhoff/Dropbox/proj_multieyetem
#
data_path = "/home/stefanappelhoff/Dropbox/proj_multieyetem"

# We only work on a single participant for now
participant_id = "1L974Q"


