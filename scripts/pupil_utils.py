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
    vt_start_multiplyer = 8,
    vt_end_multiplyer = 8,
    low_v_thresh_multiplyer = 1.5,
    max_search_duration=0.5
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


        # clipping the top 1 percent of data to not have outliers influence the threshold
        velocity_clipped = mstats.winsorize(velocity, limits=[0.01, 0.01])  # clip top and bottom 1%
        abs_velocity = np.abs(velocity_clipped.data)

        # now using median absolute deviation
        med = np.nanmedian(abs_velocity)
        # median absolute deviation
        mad = np.nanmedian(np.abs(abs_velocity - med))

        vt_start = med + vt_start_multiplyer * mad
        vt_end = med + vt_end_multiplyer * mad
        low_v_thresh = med * low_v_thresh_multiplyer

        i = 0
        while i < len(velocity) - 1:
            found_dilation = False
            found_end = False
            
            # Blink onset: rapid constriction
            if velocity[i] < -vt_start:
                istart = i
                
                # Limit search range
                max_search = int(sfreq * max_search_duration)
                
                # Dilation: search after constriction
                dilation_zone = velocity[istart:istart + max_search]
                dilation_indices = np.where(dilation_zone > vt_end)[0]
                if dilation_indices.size > 0:
                    imid = dilation_indices[0] + istart
                    found_dilation = True

                    # Look for stabilization
                    post_dilate_zone = velocity[imid:imid + max_search]
                    low_v_indices = np.where(np.abs(post_dilate_zone) < low_v_thresh)[0]
                    if low_v_indices.size > 0:
                        iend = low_v_indices[0] + imid
                        found_end = True

                        # Validate and annotate
                        duration_sec = (iend - istart) / sfreq
                        if duration_sec <= max_duration:
                            onset = (istart / sfreq) - margin_before
                            onset = max(0, onset)
                            total_duration = duration_sec + margin_before + margin_after

                            onsets.append(onset)
                            durations.append(total_duration)
                            descriptions.append(description)
                            ch_names_list.append([ch_name])

                            i = iend - 1
                            continue  # go to next blink candidate
                # Adaptive skip logic
            if found_dilation:
                i = imid + 1
            else:
                i += int(sfreq * 0.01)  # skip 10 ms
            
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
    sfreq = raw.info['sfreq']
    pre_buffer_samples = int(buffer[0] * sfreq)
    post_buffer_samples = int(buffer[1] * sfreq)

    for ch_name in channel_names:
        if ch_name not in raw.ch_names:
            continue

        ch_idx = raw.ch_names.index(ch_name)
        data = raw._data[ch_idx]

        # Find zero (or near-zero) sample indices
        nan_indices = np.where(np.isnan(data))[0]

        # Build a mask with buffer around each zero index
        mask = np.zeros_like(data, dtype=bool)
        for i in nan_indices:
            start = max(0, i - pre_buffer_samples)
            end = min(len(data), i + post_buffer_samples + 1)
            mask[start:end] = True

        # Set the masked region to NaN
        data[mask] = np.nan
    return raw


#######################
#######################

def set_zero_to_nan(raw, channel_names=('pupil_left', 'pupil_right')):
    sfreq = raw.info['sfreq']

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
            start = max(0, i)
            end = min(len(data), i + 1)
            mask[start:end] = True

        # Set the masked region to NaN
        data[mask] = np.nan
    return raw


###########################
###########################


def plot_pupil_trace_and_velocity_mad(
    raw,
    ch_name='pupil_right',
    sigma=4,
    t_start=0,
    window_duration=20,
    vt_start_multiplyer=8,
    vt_end_multiplyer=8,
    low_v_thresh_multiplyer=1.5,
    max_search_duration=0.5,
    max_duration=0.5,
):
    """
    Plot pupil trace and velocity using Median + MAD thresholding.
    Uses same logic as `annotate_velocity_blinks`.
    """

    # === Setup ===
    sfreq = raw.info['sfreq']
    t_end = t_start + window_duration
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

    # === Thresholds ===
    velocity_clipped = mstats.winsorize(velocity, limits=[0.01, 0.01])
    abs_velocity = np.abs(velocity_clipped)

    med = np.nanmedian(abs_velocity)
    mad = np.nanmedian(np.abs(abs_velocity - med))

    vt_start = med + vt_start_multiplyer * mad
    vt_end = med + vt_end_multiplyer * mad
    low_v_thresh = med * low_v_thresh_multiplyer

    # === Threshold masks (independent) ===
    below_thresh_mask = velocity < -vt_start
    above_thresh_mask = velocity > vt_end
    low_velocity_mask = np.abs(velocity) < low_v_thresh

    # === Blink detection (same as annotation logic) ===
    blink_spans = []
    i = 0
    max_search = int(sfreq * max_search_duration)

    while i < len(velocity) - 1:
        found_dilation = False
        found_end = False

        if velocity[i] < -vt_start:
            istart = i

            # Look for dilation
            dilation_zone = velocity[istart:istart + max_search]
            dilation_indices = np.where(dilation_zone > vt_end)[0]
            if dilation_indices.size > 0:
                imid = dilation_indices[0] + istart
                found_dilation = True

                # Look for low velocity
                post_dilate_zone = velocity[imid:imid + max_search]
                low_v_indices = np.where(np.abs(post_dilate_zone) < low_v_thresh)[0]
                if low_v_indices.size > 0:
                    iend = low_v_indices[0] + imid
                    found_end = True

                    duration_sec = (iend - istart) / sfreq
                    if duration_sec <= max_duration:
                        blink_spans.append((istart, iend))
                        i = iend - 1
                        continue

        # Skip ahead
        if found_dilation:
            i = imid + 1
        else:
            i += int(sfreq * 0.01)  # skip 10 ms

    # === Crop to time window ===
    mask = (times >= t_start) & (times <= t_end)
    times_crop = times[mask]
    raw_crop = data[mask]
    smooth_crop = smoothed[mask]
    velocity_crop = velocity[mask]
    below_thresh_crop = below_thresh_mask[mask]
    above_thresh_crop = above_thresh_mask[mask]
    low_velocity_crop = low_velocity_mask[mask]

    # === Plotting ===
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                            gridspec_kw={'height_ratios': [2, 1.5, 1]})

    # --- Plot 1: Pupil trace ---
    axs[0].plot(times_crop, raw_crop, label='Raw Pupil Data', color='gray', alpha=0.4)
    axs[0].plot(times_crop, smooth_crop, label=f'Smoothed (σ={sigma})', color='blue')

    for istart, iend in blink_spans:
        if t_start * sfreq <= istart <= t_end * sfreq:
            axs[0].axvspan(times[istart], times[iend], color='red', alpha=0.3)

    axs[0].legend(['Raw Pupil Data', f'Smoothed (σ={sigma})', 'Detected Blink'])
    axs[0].set_ylabel('Pupil Size')
    axs[0].set_title(f'{ch_name} — Pupil Trace and Velocity (Time: {t_start}-{t_end}s)')
    axs[0].grid(True)

    # --- Plot 2: Velocity trace ---
    axs[1].plot(times_crop, velocity_crop, label='Velocity', color='green')
    axs[1].axhline(vt_end, color='purple', linestyle='--', linewidth=1.5, label=f'vt_end = {vt_end:.2f}')
    axs[1].axhline(-vt_start, color='orange', linestyle='--', linewidth=1.5, label=f'-vt_start = {-vt_start:.2f}')
    axs[1].legend()
    axs[1].set_ylabel('Velocity')
    axs[1].grid(True)

    # --- Plot 3: Independent threshold masks ---
    axs[2].plot(times_crop, below_thresh_crop.astype(int), drawstyle='steps-mid', color='orange', label='velocity < -vt_start')
    axs[2].plot(times_crop, above_thresh_crop.astype(int) + 1.2, drawstyle='steps-mid', color='purple', label='velocity > vt_end')
    axs[2].plot(times_crop, low_velocity_crop.astype(int) + 2.4, drawstyle='steps-mid', color='black', label='|velocity| < low_v_thresh')

    axs[2].set_yticks([0, 1.2, 2.4])
    axs[2].set_yticklabels(['< -vt_start', '> vt_end', 'low velocity'])
    axs[2].set_ylim(-0.5, 3.5)
    axs[2].set_xlabel('Time (s)')
    axs[2].set_title('All Threshold Events (Independent of Blink Detection)')
    axs[2].legend(loc='upper right')
    axs[2].grid(True)

    fig.suptitle(f'{filename if filename else ""}')
    plt.tight_layout()
    plt.show()

    # === Print thresholds for logging ===
    print(f'Median: {med:.3f}, MAD: {mad:.3f}')
    print(f'vt_start (med + {vt_start_multiplyer}×MAD): {vt_start:.3f}')
    print(f'vt_end (med + {vt_end_multiplyer}×MAD): {vt_end:.3f}')
    print(f'low_v_thresh (med * {low_v_thresh_multiplyer}): {low_v_thresh:.3f}')





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
        "Linear": _interp_ignore_nan(np.interp),
        "CubicSpline": _interp_ignore_nan(sp.interpolate.CubicSpline),
        "MonotonicCubicSpline": _interp_ignore_nan(sp.interpolate.PchipInterpolator),
        "NonCubicSpline": _interp_ignore_nan(sp.interpolate.make_interp_spline),
        "Nearest": _interp_ignore_nan(sp.interpolate.interp1d),
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
    
        if len(blink_indices) == 0 or len(non_blink_indices) < 2:
            continue

        # Interpolation
        interp_func = interp_funcs[method]
        interpolated_samples = interp_func(
            raw.times[blink_indices],
            raw.times[non_blink_indices],
            raw._data[ci, non_blink_indices]
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


def _interp_ignore_nan(interp_class_or_func):
    """Wrap an interpolation class/function to safely skip NaNs in fp."""

    def wrapped(x, xp, fp):
        valid = ~np.isnan(fp)
        if np.sum(valid) < 2:
            return np.full_like(x, np.nan)  # Not enough data
        xp_valid = xp[valid]
        fp_valid = fp[valid]

        # Handle function vs. class-based interpolators
        if callable(interp_class_or_func) and hasattr(interp_class_or_func, "__call__"):
            try:
                interpolator = interp_class_or_func(xp_valid, fp_valid)
                return interpolator(x)
            except Exception:
                # fallback for np.interp
                return np.interp(x, xp_valid, fp_valid)
        else:
            return np.interp(x, xp_valid, fp_valid)

    return wrapped




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


def get_annotations_per_epoch_custom(self):
    """Get a list of annotations that occur during each epoch.

    Returns
    -------
    epoch_annots : list
        A list of lists (with length equal to number of epochs) where each
        inner list contains any annotations that overlap the corresponding
        epoch. Annotations are stored as a :class:`tuple` of onset,
        duration, description (not as a :class:`~mne.Annotations` object),
        where the onset is now relative to time=0 of the epoch, rather than
        time=0 of the original continuous (raw) data.
    """
    # create a list of annotations for each epoch
    epoch_annot_list = [[] for _ in range(len(self.events))]

    # check if annotations exist
    if self.annotations is None:
        return epoch_annot_list

    # when each epoch and annotation starts/stops
    # no need to account for first_samp here...
    epoch_tzeros = self.events[:, 0] / self._raw_sfreq
    epoch_starts, epoch_stops = (
        np.atleast_2d(epoch_tzeros) + np.atleast_2d(self.times[[0, -1]]).T
    )
    # ... because first_samp isn't accounted for here either
    annot_starts = self._annotations.onset
    annot_stops = annot_starts + self._annotations.duration

    # the first two cases (annot_straddles_epoch_{start|end}) will both
    # (redundantly) capture cases where an annotation fully encompasses
    # an epoch (e.g., annot from 1-4s, epoch from 2-3s). The redundancy
    # doesn't matter because results are summed and then cast to bool (all
    # we care about is presence/absence of overlap).
    annot_straddles_epoch_start = np.logical_and(
        np.atleast_2d(epoch_starts) >= np.atleast_2d(annot_starts).T,
        np.atleast_2d(epoch_starts) < np.atleast_2d(annot_stops).T,
    )

    annot_straddles_epoch_end = np.logical_and(
        np.atleast_2d(epoch_stops) > np.atleast_2d(annot_starts).T,
        np.atleast_2d(epoch_stops) <= np.atleast_2d(annot_stops).T,
    )

    # this captures the only remaining case we care about: annotations
    # fully contained within an epoch (or exactly coextensive with it).
    annot_fully_within_epoch = np.logical_and(
        np.atleast_2d(epoch_starts) <= np.atleast_2d(annot_starts).T,
        np.atleast_2d(epoch_stops) >= np.atleast_2d(annot_stops).T,
    )

    # combine all cases to get array of shape (n_annotations, n_epochs).
    # Nonzero entries indicate overlap between the corresponding
    # annotation (row index) and epoch (column index).
    all_cases = (
        annot_straddles_epoch_start
        + annot_straddles_epoch_end
        + annot_fully_within_epoch
    )

    # for each Epoch-Annotation overlap occurrence:
    for annot_ix, epo_ix in zip(*np.nonzero(all_cases)):
        this_annot = self._annotations[annot_ix]
        this_tzero = epoch_tzeros[epo_ix]
        # adjust annotation onset to be relative to epoch tzero...
        annot = (
            this_annot["onset"] - this_tzero,
            this_annot["duration"],
            this_annot["description"],
            this_annot["ch_names"],
            epo_ix
        )
        # ...then add it to the correct sublist of `epoch_annot_list`
        epoch_annot_list[epo_ix].append(annot)
    return epoch_annot_list