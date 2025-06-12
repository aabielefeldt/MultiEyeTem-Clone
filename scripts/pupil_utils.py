import numpy as np
import mne
from mne import Annotations


def annotate_velocity_blinks(
    raw,
    channel_names=('pupil_left', 'pupil_right'),
    velocity_thresh=None,
    velocity_percentile=99.5,
    margin_before=0.05,
    margin_after=0.05,
    description='BAD_velo_blink',
    sfreq=None
):
    """
    Annotate blink-like events based on high pupil velocity in individual eye channels.

    Parameters
    ----------
    raw : mne.io.Raw
        The Raw object containing pupil data.
    channel_names : tuple of str
        Names of the pupil channels (e.g., ('pupil_left', 'pupil_right')).
    velocity_thresh : float | None
        Fixed velocity threshold above which to detect blinks. If None, uses `velocity_percentile`.
    velocity_percentile : float
        Percentile of velocity used to compute adaptive threshold if `velocity_thresh` is None.
    margin_before : float
        Time in seconds to include before the detected blink onset.
    margin_after : float
        Time in seconds to include after the detected blink offset.
    description : str
        Label to assign to the blink annotations.
    sfreq : float | None
        Sampling frequency. If None, uses `raw.info['sfreq']`.

    Returns
    -------
    raw : mne.io.Raw
        The same Raw object with additional blink annotations associated with individual channels.
    """
    if sfreq is None:
        sfreq = raw.info['sfreq']

    # does this line have to go in the loop to make channel specific annotations?
    new_onsets, new_durations, new_descriptions, new_ch_names = [], [], [], []

    for ch_name in channel_names:
        if ch_name not in raw.ch_names:
            raise ValueError(f"Channel '{ch_name}' not found in raw data.")
        
        # Get channel data
        data = raw.copy().pick(ch_name).get_data()[0]
        velocity = np.gradient(data) * sfreq

        # Compute threshold
        if velocity_thresh is None:
            thresh = np.percentile(velocity, velocity_percentile)
            print(f"Computed velocity threshold for {ch_name}: {thresh}")
        else:
            thresh = velocity_thresh

        # Detect high-velocity segments
        above_thresh = velocity > thresh

        above_thresh_constriction = velocity < -thresh  # negative slope
        above_thresh_dilation = velocity > thresh   


        blink_onsets, blink_offsets = [], []

        in_blink = False

        for i, is_above in enumerate(above_thresh):
            if is_above and not in_blink:
                in_blink = True
                blink_onsets.append(i)
            elif not is_above and in_blink:
                in_blink = False
                blink_offsets.append(i)
        
        if in_blink and len(blink_onsets) > len(blink_offsets):
            blink_offsets.append(len(data) - 1)

        # Convert to annotations
        for onset_idx, offset_idx in zip(blink_onsets, blink_offsets):
            onset_sec = max(0, (onset_idx / sfreq) - margin_before)
            duration = ((offset_idx - onset_idx) / sfreq) + margin_before + margin_after
            new_onsets.append(onset_sec)
            new_durations.append(duration)
            new_descriptions.append(description)
            new_ch_names.append([ch_name])  # Associate with only this channel

    # Create and assign new annotations
    if new_onsets:
        new_annotations = Annotations(
            onset=new_onsets,
            duration=new_durations,
            description=new_descriptions,
            orig_time=raw.annotations.orig_time,
            ch_names=new_ch_names
        )
        raw.set_annotations(raw.annotations + new_annotations)
    else:
        print("No velocity-based blinks detected.")

    return raw



def plot_pupil_and_velocity(raw, pupil_channel, sfreq=None):
    """
    Plot raw pupil data and its velocity (1st derivative) for inspection.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw object with pupil data.
    pupil_channel : str
        Name of the channel (e.g., 'pupil_right').
    sfreq : float or None
        Sampling frequency. If None, uses raw.info["sfreq"].
    """
    import matplotlib.pyplot as plt

    if sfreq is None:
        sfreq = raw.info["sfreq"]

    data = raw.copy().pick(pupil_channel).get_data()[0]
    times = raw.times
    velocity = np.gradient(data) * sfreq

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axs[0].plot(times, data, label="Pupil size")
    axs[0].set_ylabel("Pupil size")
    axs[0].legend()

    axs[1].plot(times, velocity, label="Velocity", color="orange")
    axs[1].axhline(y=0, color="grey", linestyle="--", linewidth=0.5)
    axs[1].set_ylabel("Velocity")
    axs[1].set_xlabel("Time (s)")
    axs[1].legend()

    plt.tight_layout()
    plt.show()
