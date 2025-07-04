import numpy as np
import mne
from mne import Annotations
from scipy.ndimage import gaussian_filter1d
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from mne.annotations import Annotations, _annotations_starts_stops
from mne.preprocessing.artifact_detection import _annotations_from_mask
from mne._fiff.constants import FIFF
from mne.utils import logger, _check_preload, _validate_type, warn # pylint: disable=no-name-in-module
from mne.io import BaseRaw # pylint: disable=no-name-in-module
from mne.io.eyelink.eyelink import RawEyelink
from scipy.signal import welch
from scipy.stats import mstats
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d


def annotate_velocity_blinks(
    raw,
    channel_names=('pupil_left', 'pupil_right'),
    margin_before=0.01,
    margin_after=0.01,
    max_duration=0.5,
    description='BAD_velo_blink',
    sfreq=None,
    sigma=4,
    vt_start_multiplyer = 6,
    vt_end_multiplyer = 6,
    low_v_thresh_devider = 4
):
    """
    Annotate blinks based on pupil velocity profile:
    fast constriction -> fast dilation -> low velocity.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw object with pupil data.
    channel_names : tuple of str
        Names of pupil channels (e.g., ('pupil_left', 'pupil_right')).
    velocity_percentile : float
        Percentile to determine velocity threshold adaptively.
    margin_before : float
        Time (in seconds) to extend before blink onset.
    margin_after : floatx
        Time (in seconds) to extend after blink offset.
    max_duration : float
        Maximum allowable blink duration (in seconds).
    description : str
        Annotation label.
    sfreq : float | None
        Sampling frequency. If None, uses raw.info['sfreq'].
    sigma : float
        Sigma for Gaussian smoothing.

    Returns
    -------
    raw : mne.io.Raw
        Modified Raw object with blink annotations.
    """
    annotations = raw.annotations

    if sfreq is None:
        sfreq = raw.info['sfreq']

    onsets, durations, descriptions, ch_names_list = [], [], [], []

    for ch_name in channel_names:
        if ch_name not in raw.ch_names:
            raise ValueError(f"Channel '{ch_name}' not found in raw.")

        data = raw.copy().pick(ch_name).get_data()[0]

        if sigma>0:
            smoothed = gaussian_filter1d(data, sigma=sigma)
            velocity = np.gradient(smoothed) * sfreq
        else:
            velocity = np.gradient(data) * sfreq
        # smoothed = gaussian_filter1d(data, sigma=sigma)
        # velocity = np.gradient(smoothed) * sfreq

        # clipping the top 1 percent of data to not have outliers influence the threshold
        velocity_clipped = mstats.winsorize(velocity, limits=[0.01, 0.01])  # clip top and bottom 1%
        abs_velocity = np.abs(velocity_clipped.data)

        # now using median absolute deviation
        med = np.nanmedian(abs_velocity)
        # median absolute deviation
        mad = np.nanmedian(np.abs(abs_velocity - med))

        vt_start = med + vt_start_multiplyer * mad
        vt_end = med + vt_end_multiplyer * mad

        above, below, close_zero = [], [], []

        i = 0
        while i < len(velocity) - 1:
            
            # Blink onset: rapid constriction
            if velocity[i] < -vt_start:
                istart = i
                
                # Dilation: search after constriction
                dilation_zone = velocity[istart:]
                try:
                    imid = np.where(dilation_zone > vt_end)[0][0] + istart
                except IndexError:
                    break  # No dilation found, stop

                # Stabilization: velocity drops close to zero
                post_dilate_zone = velocity[imid:]
                low_v_thresh = mad / low_v_thresh_devider
                try:
                    iend = np.where(np.abs(post_dilate_zone) < low_v_thresh)[0][0] + imid
                    
                except IndexError:
                    break  # No low-velocity end found

                # Check duration
                duration_sec = (iend - istart) / sfreq
                if duration_sec <= max_duration:
                    onset = (istart / sfreq) - margin_before
                    onset = max(0, onset)
                    total_duration = duration_sec + margin_before + margin_after

                    onsets.append(onset)
                    durations.append(total_duration)
                    descriptions.append(description)
                    ch_names_list.append([ch_name])
                    i = iend + 1
                else:
                    i+=1
            else:
                i += 1
            
    print(f"{ch_name}: med={med:.2f}, mad={mad:.2f}, vt_start={vt_start:.2f}, vt_end={vt_end:.2f}, low_v_thresh={low_v_thresh:.2f}")


    if onsets:
        new_annotations = Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions,
            orig_time=raw.annotations.orig_time,
            ch_names=ch_names_list
        )
        raw.set_annotations(annotations + new_annotations)
    else:
        print("No blinks detected.")

    return raw

#######################
#######################

def remove_missing_data(raw, channel_names=('pupil_left', 'pupil_right'), buffer=[0.15, 0.15]):
    raw = raw.copy()  
    sfreq = raw.info['sfreq']
    pre_buffer_samples = int(buffer[0] * sfreq)
    post_buffer_samples = int(buffer[1] * sfreq)

    for ch_name in channel_names:
        if ch_name not in raw.ch_names:
            continue

        ch_idx = raw.ch_names.index(ch_name)
        data = raw._data[ch_idx]

        # Find zero (or near-zero) sample indices
        zero_indices = np.where(np.isclose(data, 0, atol=1e-8))[0]

        # Build a mask with buffer around each zero index
        mask = np.zeros_like(data, dtype=bool)
        for i in zero_indices:
            start = max(0, i - pre_buffer_samples)
            end = min(len(data), i + post_buffer_samples + 1)
            mask[start:end] = True

        # Set the masked region to NaN
        data[mask] = np.nan
    return raw

###########################
###########################


def plot_pupil_trace_and_velocity_mad(
    raw,
    ch_name='pupil_left',
    sigma=4,
    t_start=0,
    window_duration=20,
    mad_start_mult=6,
    mad_end_mult=6,
    low_v_thresh_devider=0.175,
    max_duration=0.5
):
    """
    Plot pupil trace and velocity using Median + MAD thresholding.
    Highlights blink onsets, dilation peaks, and stabilization.
    """
    t_end = t_start + window_duration
    sfreq = raw.info['sfreq']
    times = raw.times
    data = raw.copy().pick(ch_name).get_data()[0]
    filename = raw.filenames

    # === Smooth and compute velocity ===
    if sigma > 0:
        smoothed = gaussian_filter1d(data, sigma=sigma)
        velocity = np.gradient(smoothed) * sfreq
    else:
        smoothed = data
        velocity = np.gradient(data) * sfreq

    # === Compute thresholds ===
    velocity_clipped = mstats.winsorize(velocity, limits=[0.01, 0.01])
    abs_velocity = np.abs(velocity_clipped)

    med = np.nanmedian(abs_velocity)
    mad = np.nanmedian(np.abs(abs_velocity - med))

    vt_start = med + mad_start_mult * mad
    vt_end = med + mad_end_mult * mad

    # === Detect blink-related events ===
    above = np.zeros_like(velocity, dtype=bool)
    below = np.zeros_like(velocity, dtype=bool)
    close_zero = []

    i = 0
    while i < len(velocity) - 1:
        if velocity[i] < -vt_start:
            istart = i

            dilation_zone = velocity[istart:]
            try:
                imid = np.where(dilation_zone > vt_end)[0][0] + istart
            except IndexError:
                break

            post_dilate_zone = velocity[imid:]
            low_v_thresh = mad / low_v_thresh_devider
            try:
                iend = np.where(np.abs(post_dilate_zone) < low_v_thresh)[0][0] + imid
            except IndexError:
                break

            duration_sec = (iend - istart) / sfreq
            if duration_sec <= max_duration:
                above[istart] = True
                below[imid] = True
                close_zero.append(iend)

                i = iend + 1
            else:
                i+=1
        else:
            i += 1

    # === Crop to time window ===
    mask = (times >= t_start) & (times <= t_end)
    times_crop = times[mask]
    raw_crop = data[mask]
    smooth_crop = smoothed[mask]
    velocity_crop = velocity[mask]
    above_crop = above[mask]
    below_crop = below[mask]

    # Convert close_zero indices to times and crop them
    close_zero_times = times[close_zero]
    close_zero_crop = close_zero_times[(close_zero_times >= t_start) & (close_zero_times <= t_end)]

    # === Plot ===
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # --- Plot pupil trace ---
    axs[0].plot(times_crop, raw_crop, label='Raw Pupil Data', color='gray', alpha=0.4)
    axs[0].plot(times_crop, smooth_crop, label=f'Smoothed (σ={sigma})', color='blue')

    #  Keep track of which labels have been added
    label_flags = {'above': False, 'below': False, 'close_zero': False}

    # Blink Start
    for t in times_crop[above_crop]:
        label = 'Blink Start' if not label_flags['above'] else ""
        axs[0].axvline(t, color='orange', linestyle='--', alpha=0.7, label=label)
        label_flags['above'] = True

    # Dilation Peak
    for t in times_crop[below_crop]:
        label = 'Dilation Peak' if not label_flags['below'] else ""
        axs[0].axvline(t, color='purple', linestyle='--', alpha=0.7, label=label)
        label_flags['below'] = True

    # Stabilization
    for t in close_zero_crop:
        label = 'Stabilization' if not label_flags['close_zero'] else ""
        axs[0].axvline(t, color='black', linestyle=':', alpha=0.7, label=label)
        label_flags['close_zero'] = True

    axs[0].set_ylabel('Pupil Size')
    axs[0].set_title(f'{ch_name} — Pupil Trace and Velocity (Time: {t_start}-{t_end}s)')
    axs[0].legend(loc='upper right')
    axs[0].grid(True)

    # --- Plot velocity ---
    axs[1].plot(times_crop, velocity_crop, label='Velocity', color='green')
    axs[1].axhline(vt_end, color='red', linestyle='--', linewidth=1.5, label=f'+Thresh (vt_end) = {vt_end:.1f}')
    axs[1].axhline(-vt_start, color='red', linestyle='--', linewidth=1.5, label=f'-Thresh (vt_start) = {-vt_start:.1f}')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Velocity')
    axs[1].legend()
    axs[1].grid(True)

    fig.suptitle(f'{filename[0] if filename else ""}')
    plt.tight_layout()
    plt.show()

    print(f'Median: {med:.3f}, MAD: {mad:.3f}')
    print(f'vt_start (med + {mad_start_mult}×MAD): {vt_start:.3f}')
    print(f'vt_end (med + {mad_end_mult}×MAD): {vt_end:.3f}')

    return above, below, close_zero





#######################
# Felix interpolation function
# looks perfect to me so far
# need to check that it works on my specific annotations (and change those in match maybe)

def interpolate_blinks(raw, buffer=0.05, match="BAD_blink", interpolate_gaze=False, method="CubicSpline"):
    """Interpolate eyetracking signals during blinks.

    This function uses the timing of blink annotations to estimate missing
    data. Missing values are then interpolated linearly. Operates in place.

    Parameters
    ----------
    raw : instance of Raw
        The raw data with at least one ``'pupil'`` or ``'eyegaze'`` channel.
    buffer : float | array-like of float, shape ``(2,))``
        The time in seconds before and after a blink to consider invalid and
        include in the segment to be interpolated over. Default is ``0.05`` seconds
        (50 ms). If array-like, the first element is the time before the blink and the
        second element is the time after the blink to consider invalid, for example,
        ``(0.025, .1)``.
    match : str | list of str
        The description of annotations to interpolate over. If a list, the data within
        all annotations that match any of the strings in the list will be interpolated
        over. If a ``match`` starts with ``'BAD_'``, that part will be removed from the
        annotation description after interpolation. Defaults to ``'BAD_blink'``.
    interpolate_gaze : bool
        If False, only apply interpolation to ``'pupil channels'``. If True, interpolate
        over ``'eyegaze'`` channels as well. Defaults to False, because eye position can
        change in unpredictable ways during blinks.
    method : str
        The 1d interpolation method that is used on the annotations. Default is ``Linear```
        to use linear interpolation. Other 1d methods are listed at
        https://docs.scipy.org/doc/scipy/tutorial/interpolate.html

    Returns
    -------
    self : instance of Raw
        Returns the modified instance.

    Notes
    -----
    This function is copied from mne.preprocessing.eyetracking and adapted to our use case, which
    is interpolating not specifically annotated blink intervals, but all bad segments that we can
    specify with annotations.
    For example, segments of data can be set to NaN based on any criteria. Annotations for the NaN
    segments can then be interpolated.

    If this function works well, it would be worth contributing that to MNE as well

    .. versionadded:: 1.5
    """

    _check_preload(raw, "interpolate_blinks")
    _validate_type(raw, BaseRaw, "raw")
    _validate_type(buffer, (float, tuple, list, np.ndarray), "buffer")
    _validate_type(match, (str, tuple, list, np.ndarray), "match")

    # determine the buffer around blinks to include in the interpolation
    buffer = np.array(buffer, dtype=float)
    if buffer.size == 1:
        buffer = np.array([buffer, buffer])

    if isinstance(match, str):
        match = [match]

    # get the blink annotations
    blink_annots = [annot for annot in raw.annotations if annot["description"] in match]
    if not blink_annots:
        warn(f"No annotations matching {match} found. Aborting.")
        return raw
    _interpolate_blinks(
        raw, buffer, blink_annots, match, interpolate_gaze=interpolate_gaze, method=method
    )

    # remove bad from the annotation description
    for desc in match:
        if desc.startswith("BAD_"):
            logger.info(f"Removing 'BAD_' from {desc} and add '_interp'.")
            raw.annotations.rename({desc: desc.replace("BAD_", "") + "_interp"})
    return raw


def _interpolate_blinks(raw, buffer, blink_annots, match, interpolate_gaze, method="Linear"):
    """Interpolate eyetracking signals during blinks in-place."""

    # function handles
    interp_funcs = {
        "Linear" : np.interp,
        "CubicSpline" : _interp_cubicspline,
        "MonotonicCubicSpline" : _interp_pchip,
        "NonCubicSpline" : _interp_make_spline,
        "Nearest" : _interp_1d,
    }

    logger.info("Interpolating missing data during blinks...")
    pre_buffer, post_buffer = buffer
    # iterate over each eyetrack channel and interpolate the blinks
    interpolated_chs = []
    for ci, ch_info in enumerate(raw.info["chs"]):
        if interpolate_gaze:  # interpolate over all eyetrack channels
            if ch_info["kind"] != FIFF.FIFFV_EYETRACK_CH:
                continue
        else:  # interpolate over pupil channels only
            if ch_info["coil_type"] != FIFF.FIFFV_COIL_EYETRACK_PUPIL:
                continue
        # Create an empty boolean mask
        mask = np.zeros_like(raw.times, dtype=bool)
        starts, ends = _annotations_starts_stops(raw, match)
        starts = np.divide(starts, raw.info["sfreq"])
        ends = np.divide(ends, raw.info["sfreq"])
        for annot, start, end in zip(blink_annots, starts, ends):
            if "ch_names" not in annot or not annot["ch_names"]:
                msg = f"Blink annotation missing values for 'ch_names' key: {annot}"
                raise ValueError(msg)
            start -= pre_buffer
            end += post_buffer
            if ch_info["ch_name"] not in annot["ch_names"]:
                continue  # skip if the channel is not in the blink annotation
            # Update the mask for times within the current blink period
            mask |= (raw.times >= start) & (raw.times <= end)
        blink_indices = np.where(mask)[0]
        non_blink_indices = np.where(~mask)[0]

        # Interpolation
        interpolated_samples = interp_funcs[method](
            raw.times[blink_indices],
            raw.times[non_blink_indices],
            raw._data[ci, non_blink_indices],
        )
        # Replace the samples at the blink_indices with the interpolated values
        raw._data[ci, blink_indices] = interpolated_samples
        interpolated_chs.append(ch_info["ch_name"])
    if interpolated_chs:
        logger.info(
            f"Interpolated {len(interpolated_chs)} channels: {interpolated_chs}"
        )
    else:
        warn("No channels were interpolated.")


def _interp_cubicspline(x, xp, fp):
    """wrapper for scipy interpolation classes"""
    Interp = sp.interpolate.CubicSpline(xp, fp)
    return Interp(x)


def _interp_pchip(x, xp, fp):
    """wrapper for scipy interpolation classes"""
    Interp = sp.interpolate.PchipInterpolator(xp, fp)
    return Interp(x)


def _interp_make_spline(x, xp, fp):
    """wrapper for scipy interpolation classes"""
    Interp = sp.interpolate.make_interp_spline(xp, fp)
    return Interp(x)


def _interp_1d(x, xp, fp):
    """wrapper for scipy interpolation classes"""
    Interp = sp.interpolate.interp1d(xp, fp)
    return Interp(x)



# %%
def merge_and_rename_annotations(
    raw,
    match=["BAD_velo_blink"],
    min_gap_sec=0.1,
    new_description="BAD_merged_blink",
    rename_original_to="blink_used_for_merged",
    channel_names=['pupil_right', 'pupil_left']
):
    """
    Merge close annotations per channel and rename originals used in merge.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw object with annotations.
    match : list of str
        Descriptions of annotations to consider for merging.
    min_gap_sec : float
        Maximum time between annotations to allow merging.
    new_description : str
        Description for newly merged annotations.
    rename_original_to : str
        New description for original annotations used in merges.
    channel_names : list of str
        Names of channels to process separately.

    Returns
    -------
    raw : mne.io.Raw
        Raw object with updated annotations.
    """

    original = raw.annotations
    all_new_annots = []
    rename_indices = []

    for ch in channel_names:
        # Filter annotations by channel name and matching descriptions
        relevant_idxs = [
            i for i, ann in enumerate(original)
            if ann["description"] in match and (ann["ch_names"] is not None and ch in ann["ch_names"])
        ]

        if not relevant_idxs:
            continue

        # Sort by onset
        sorted_idxs = sorted(relevant_idxs, key=lambda i: original.onset[i])

        # Group close annotations
        groups = []
        current = []

        for idx in sorted_idxs:
            if not current:
                current = [idx]
                continue

            last = current[-1]
            last_end = original.onset[last] + original.duration[last]
            this_start = original.onset[idx]

            if this_start - last_end <= min_gap_sec:
                current.append(idx)
            else:
                groups.append(current)
                current = [idx]

        if current:
            groups.append(current)

        # Process groups (only those with more than 1 annotation)
        for group in groups:
            if len(group) < 2:
                continue  # skip groups with only one annotation

            group_onsets = original.onset[group]
            group_durations = original.duration[group]
            start = np.min(group_onsets)
            end = np.max(group_onsets + group_durations)

            all_new_annots.append({
                "onset": start,
                "duration": end - start,
                "description": new_description,
                "ch_names": [ch]
            })

            rename_indices.extend(group)

    # Rename original annotations used in merges
    all_old_descriptions = list(raw.annotations.description)
    all_ch_names = list(raw.annotations.ch_names)
    if rename_indices:
        for i in rename_indices:
            all_old_descriptions[i] = rename_original_to
    
    rename_annots = mne.Annotations(
        onset=raw.annotations.onset,
        duration=raw.annotations.duration,
        description=all_old_descriptions,
        orig_time=raw.annotations.orig_time,
        ch_names=all_ch_names # probably unnecessary change
        )
    raw.set_annotations(rename_annots)

    # Add merged annotations
    if all_new_annots:
        merged = mne.Annotations(
            onset=[a["onset"] for a in all_new_annots],
            duration=[a["duration"] for a in all_new_annots],
            description=[a["description"] for a in all_new_annots],
            ch_names=[a["ch_names"] for a in all_new_annots],
            orig_time=original.orig_time
        )
        raw.set_annotations(raw.annotations + merged)

    return raw
# %%