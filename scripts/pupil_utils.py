import numpy as np
import mne
from mne import Annotations
from scipy.ndimage import gaussian_filter1d
import scipy as sp
import pandas as pd

from mne.annotations import Annotations, _annotations_starts_stops
from mne.preprocessing.artifact_detection import _annotations_from_mask
from mne._fiff.constants import FIFF
from mne.utils import logger, _check_preload, _validate_type, warn # pylint: disable=no-name-in-module
from mne.io import BaseRaw # pylint: disable=no-name-in-module
from mne.io.eyelink.eyelink import RawEyelink
from scipy.signal import welch

def annotate_velocity_blinks(
    raw,
    channel_names=('pupil_left', 'pupil_right'),
    velocity_percentile=99.5,
    margin_before=0.01,
    margin_after=0.01,
    max_duration=0.5,
    description='BAD_velo_blink',
    sfreq=None,
    sigma=4
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
        smoothed = gaussian_filter1d(data, sigma=sigma)
        velocity = np.gradient(smoothed) * sfreq

        vt_start = np.percentile(np.abs(velocity), velocity_percentile)
        vt_end = vt_start / 2

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
                low_v_thresh = np.nanstd(velocity) / 100
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

                # Move index to end of blink to avoid overlapping blinks
                i = iend + 1
            else:
                i += 1

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

def remove_missing_data(raw, channel_names=('pupil_left', 'pupil_right'), buffer=[0.05, 0.05]):
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

