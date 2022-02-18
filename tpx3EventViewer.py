#!/usr/bin/env python3
import _tkinter
import math
import multiprocessing
from functools import partial

from matplotlib.ticker import EngFormatter
from scipy import fftpack
import argparse
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import mrcfile
from matplotlib.widgets import Slider, CheckButtons
from matplotlib import animation, patches
from PIL import Image
import os
import copy
import netpbmfile

import tqdm

from gaussians import event_gaussian, get_gauss_distribution

VERSION = '2.1.0'
# The Timepix3 fine ToA clock is 640 Mhz. This is equal to a tick length of 1.5625 ns.
tpx3_tick = 1.5625e-09
ticks_second = 1. / tpx3_tick

plt.rcParams.update({
    "font.size": 12,
    "font.family": 'sans-serif',
    "svg.fonttype": 'none'
})


def main():
    settings = parse_arguments()

    f = h5py.File(settings.FILE, 'r')

    if settings.cluster_stats:
        if 'clusters' not in f:
            print("ERROR: No /cluster dataset present in file (%s)." % settings.FILE)
            return 1

        print_cluster_stats(f['clusters'], settings.cluster_stats_tot, settings.cluster_stats_size)
        return 0

    # Get source
    if settings.hits:
        source = 'hits'

        if source not in f:
            print(
                "ERROR: No /hits dataset present in file (%s). Did you mean without --hits to display events?" % settings.FILE)
            return 1
    else:
        source = 'events'

        if source not in f:
            print(
                "ERROR: No /events dataset present in file (%s). Did you mean to use --hits to display hits?" % settings.FILE)
            return 1

    # Output filename
    filename = ''
    if settings.t or settings.m:
        if settings.f:
            filename = settings.f
        else:
            ext = '.tif' if settings.t else '.mrc'
            filename = os.path.splitext(os.path.basename(settings.FILE))[0] + ext

        if os.path.exists(filename) and not settings.o:
            print("ERROR: Output file %s already exists, and overwrite not specified" % filename)
            return 1

    data = f[source]

    if data.attrs['version'] != VERSION:
        print("WARNING: Version of data file does not match version of tpx3EventViewer (%s vs %s)" % (
            data.attrs['version'], VERSION))

    # Get z_source
    z_source = None
    if settings.hits_tot:
        z_source = 'ToT'
    elif settings.hits_toa:
        z_source = 'ToA'
    elif settings.events_sumtot:
        z_source = 'sumToT'
    elif settings.events_nhits:
        z_source = 'nHits'

    # Get shape of matrix
    shape = data.attrs['shape']

    # Load data and apply ToT threshold
    if settings.hits and (settings.tot_threshold > 0 or settings.tot_limit < 1023):
        data = data[()]

        if settings.tot_threshold > 0:
            data = data[data['ToT'] > settings.tot_threshold]

        if settings.tot_limit < 1023:
            data = data[data['ToT'] < settings.tot_limit]
    else:
        data = data[()]

    # Filter to requested chip
    if settings.chip is not None:
        data = data[data['chipId'] == settings.chip]

    if len(data) == 0:
        print("ERROR: No hits or events present (after filtering). This would result in an empty frame.")
        return 1

    # Determine frame indices
    frames_idx = calculate_frames_idx(data, settings.exposure, settings.start, settings.end)

    if settings.timing_stats:
        min_toa = f[source].attrs.get('min_toa', -1)
        max_toa = f[source].attrs.get('max_toa', -1)
        timing_stats(data, frames_idx, min_toa, max_toa, settings.n)
        return 0

    # Gain and defect (dead) pixel settings
    gain = None
    defects = None
    if settings.gain:
        with h5py.File(settings.gain, mode='r') as h5:
            if h5.attrs['shape'] != shape:
                print("ERROR: Shape matrix stored in gain file does not match image shape.")
                return 1

            gain = h5['gain'][()]

            if settings.correct_defect_pixels:
                defects = h5['defects'][()]

    # Calculate all frames
    frames = list()
    if not settings.gauss:
        for frame_idx in frames_idx:
            raw_frame = to_frame(frame_idx['d'], z_source, shape, settings.super_res, settings.normalize)
            frames.append(frame_modifications(raw_frame, settings.rotation, settings.flip_x, settings.flip_y,
                                              settings.power_spectrum, gain, defects))
    else:
        # TODO: Make this gaussian configurable
        raw_frames = to_frames_gaussian(frames_idx, 0.7, shape, settings.super_res)
        for raw_frame in raw_frames:
            frames.append(frame_modifications(raw_frame, settings.rotation, settings.flip_x, settings.flip_y,
                                              settings.power_spectrum, gain, defects))



    if settings.m:
        save_mrc(frames, filename)

    if settings.t:
        save_tiff(frames, settings.uint32, settings.uint8, filename)

    if not settings.n:
        show(frames, settings.animation, os.path.basename(settings.FILE))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,  # printed with -h/--help
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('FILE', help="Input .h5 file")
    parser.add_argument("-t", action='store_true', help="Store uint16 .tif file")
    parser.add_argument("--uint32", action='store_true', help="Store uint32 tif (not supported by all readers!)")
    parser.add_argument("--uint8", action='store_true', help="Store uint8 tif (supported by almost all readers)")
    parser.add_argument("-m", action='store_true', help="Store as mrc file")
    parser.add_argument("-f", metavar='FILE', help="File name for .tif file (default is .h5 file with .tif extension)")
    parser.add_argument("-o", action='store_true', help="Overwrite existing file")
    parser.add_argument("-n", action='store_true', help="Don't show interactive viewer")
    parser.add_argument("-r", "--rotation", type=int, default=0, help="Rotate 90 degrees (1: clockwise, "
                                                                      "-1 anti-clockwise, 0: none). Default: 0")
    parser.add_argument("-g", "--gain", help="MRC file with gain correction.")
    parser.add_argument("--animation", action='store_true', help="Store as animated mp4 file")
    parser.add_argument("--power_spectrum", action='store_true', help="Show power spectrum")
    parser.add_argument("--flip_x", action='store_true', help="Flip image in X")
    parser.add_argument("--flip_y", action='store_true', help="Flip image in Y")
    parser.add_argument("--hits", action='store_true', help="Use hits (default in counting mode)")
    parser.add_argument("--hits_tot", action='store_true', help="Use hits in ToT mode")
    parser.add_argument("--hits_toa", action='store_true', help="Use hits in ToA mode")
    parser.add_argument("--gauss", action='store_true',
                        help='Use events, but place back as gaussian with a certain lambda')
    parser.add_argument("--events_sumtot", action='store_true', help="Use events in sumToT mode")
    parser.add_argument("--events_nhits", action='store_true', help="Use events in nHits mode")
    parser.add_argument("--timing_stats", action='store_true', help="Show timing stats")
    parser.add_argument("--tot_threshold", type=int, default=0, help="In hits show only hits above ToT threshold")
    parser.add_argument("--tot_limit", type=int, default=1023, help="In hits show only hits below ToT limit")
    parser.add_argument("--chip", type=int, default=None, help="Limit display to certain chip")
    parser.add_argument("--normalize", action='store_true', help="Normalize to the average (useful for showing ToT)")
    parser.add_argument("--exposure", type=float, default=0, help="Max exposure time in seconds (0: infinite)")
    parser.add_argument("--start", type=float, default=0, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=0, help="End time in seconds")
    parser.add_argument("--super_res", metavar='N', type=int, default=1,
                        help="Up scale the amount of pixels by N factor")
    parser.add_argument("--cluster_stats", action='store_true', help="Show cluster stats")
    parser.add_argument("--cluster_stats_tot", type=int, default=None, help="Override cluster_stats ToT limit")
    parser.add_argument("--cluster_stats_size", type=int, default=None, help="Override cluster_stats size limit")
    parser.add_argument("--correct_defect_pixels", action='store_true', help="Correct defect pixels (supplied in gain file)")

    settings = parser.parse_args()

    if (settings.hits_tot or settings.hits_toa) and not settings.hits:
        parser.error("Need to set showing --hits when selecting --hits_tot or --hits_toa")

    return settings


def save_mrc(frames, filename):
    data = np.array(frames)

    # MRC data is saved in different orientation than tiff and imshow. We're correcting this here to be the same as tiff
    # and imshow
    data = np.flip(data, axis=(1,))

    if data.dtype == np.float64 or data.dtype == np.float32:
        print("INFO: Storing as float32")
        data = data.astype(dtype=np.float32)
    else:
        # Needs possible clipping to max uint16 values
        i16 = np.iinfo(np.uint16)
        if data.max(initial=0) > i16.max:
            print("WARNING: Cannot fit in uint16. Clipping values to uint16 max.")
            np.clip(data, 0, i16.max, data)
        print("INFO: Storing as uint16")
        data = data.astype(np.uint16)

    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(data)


def save_tiff(frames, uint32, uint8, filename):
    images = list()
    for frame in frames:
        if uint32:
            # Can store directly to image
            im = Image.fromarray(frame)
        elif uint8:
            # Needs possible clipping to max uint16 values
            i8 = np.iinfo(np.uint8)
            if frame.max() > i8.max:
                print("WARNING: Cannot fit in uint8. Clipping values to uint16 max.")
                np.clip(frame, 0, i8.max, frame)

            frame = frame.astype(dtype=np.uint8)

            im = Image.fromarray(frame)
        else:
            if frame.dtype == np.float64:
                print("INFO: Storing as float32")
                frame = frame.astype(dtype=np.float32)
                im = Image.fromarray(frame)
            else:
                # Needs possible clipping to max uint16 values
                i16 = np.iinfo(np.uint16)
                if frame.max() > i16.max:
                    print("WARNING: Cannot fit in uint16. Clipping values to uint16 max.")
                    np.clip(frame, 0, i16.max, frame)

                frame = frame.astype(dtype=np.uint16)

                im = Image.fromarray(frame)

        images.append(im)

    images[0].save(filename, save_all=True, append_images=images[1:])


def show(frames, animate, name):
    # Calculate threshold values
    frame = frames[0]
    min5 = np.percentile(frame, 5)
    min = np.min(frame)
    max95 = np.percentile(frame, 95)
    max = np.max(frame)

    if animate:
        dpi = 300
    else:
        dpi = 150

    fig = plt.figure(dpi=dpi)
    fig.canvas.manager.set_window_title(name)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # This makes pixels start at the top left, and at 0,0. This makes the most sense when compared to other
    # applications. It also matches the sub pixel positions.
    # https://matplotlib.org/stable/tutorials/intermediate/imshow_extent.html
    origin = 'upper'
    extent = [0, frame.shape[0], frame.shape[1], 0]

    im = ax.imshow(frame, vmin=min5, vmax=max95, extent=extent, origin=origin)
    ax.format_coord = lambda x, y: '%3.2f, %3.2f, %10d' % (x, y, frame[int(y), int(x)])
    fig.colorbar(im)

    def update_frame(val):
        idx = int(val)
        im.set_data(frames[idx])
        fig.canvas.draw()

    def animate_frame(idx):
        sframe.set_val(idx)
        im.set_data(frames[idx])
        fig.canvas.draw()

    def update_clim(val):
        im.set_clim([smin.val, smax.val])
        fig.canvas.draw()

    def update_check(label):
        if label == 'Grayscale':
            if im.get_cmap().name == 'gray':
                im.set_cmap(plt.rcParams['image.cmap'])
            else:
                im.set_cmap('gray')
            fig.canvas.draw()
        if label == 'foo':
            print('bar')

    axmin = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    axmax = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    rax = fig.add_axes([0.01, 0.7, 0.15, 0.15])
    check = CheckButtons(rax, ('Grayscale', 'foo'), (False, False))
    check.on_clicked(update_check)

    smin = Slider(axmin, 'Min', min, max, valinit=min5)
    smax = Slider(axmax, 'Max', min, max, valinit=max95)
    smin.on_changed(update_clim)
    smax.on_changed(update_clim)

    if len(frames) > 1:
        axframe = fig.add_axes([0.25, 0.05, 0.65, 0.03])
        sframe = Slider(axframe, 'Frame', 0, len(frames) - 1, valinit=0, valfmt="%i")
        sframe.on_changed(update_frame)

    if animate:
        # Animate
        anim = animation.FuncAnimation(fig, animate_frame, frames=len(frames) - 1, interval=1000)
        # Save
        anim.save('animation.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
    else:
        plt.show()


# This function assumes the data to be sorted on ToA!
def calculate_frames_idx(data, exposure, start_time, end_time):
    frames = list()

    toa = data['ToA']

    if start_time > 0:
        start = toa[0] + ticks_second * start_time
        start_idx = np.argmax(toa > start)

        if start_idx == 0:
            print("WARNING: Start time not found in data. Starting at time 0")
    else:
        start = toa[0]
        start_idx = 0

    if end_time > 0:
        end_final = toa[0] + ticks_second * end_time
        end_final_idx = np.argmax(toa > end_final)
        if end_final_idx == 0:
            end_final_idx = len(data) - 1
    else:
        end_final_idx = len(data) - 1

    end = start + ticks_second * exposure

    if exposure > 0:
        # Calculate all frames indices
        while start_idx < end_final_idx:
            end_idx = np.argmax(toa > end)

            # TODO: Need to be able to handle empty frames
            # if toa[end_idx] - end > exposure*0.1

            if end_idx == 0:
                end_idx = len(data) - 1

            frames.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'd': data[start_idx:end_idx]
            })

            start = end
            start_idx = end_idx
            end = start + ticks_second * exposure
    else:
        frames.append({
            'start_idx': start_idx,
            'end_idx': end_final_idx,
            'd': data[start_idx:end_final_idx]
        })

    # for frame in frames:
    #    print("Average time of frame: %f" % ((toa[frame['end_idx']] - toa[frame['start_idx']]) * tpx3_tick))

    return frames


def to_frame(frame, z_source, shape, super_resolution, normalize):
    x = frame['x']
    y = frame['y']

    if super_resolution > 1:
        x = x * super_resolution
        y = y * super_resolution
        shape = shape * super_resolution

    # By casting to int we floor the result to the bottom left pixel it was found in
    x = x.astype(dtype='uint16')
    y = y.astype(dtype='uint16')

    if z_source is None:
        data = np.ones(len(frame))
    else:
        data = frame[z_source]

    d = scipy.sparse.coo_matrix((data, (y, x)), shape=(shape, shape), dtype=np.uint32)
    f = d.todense()

    # Normalize (for example ToT average) over counts received
    if normalize:
        data = np.ones(len(frame))
        d = scipy.sparse.coo_matrix((data, (y, x)), shape=(shape, shape), dtype=np.uint32)
        n = d.todense() + 1
        f = np.divide(f, n)

    return f


def frame_modifications(f, rotation, flip_x, flip_y, power_spectrum, gain, defects):
    if gain is not None:
        f = np.multiply(f, gain)

    if defects is not None:
        f = correct_defect_pixels(f, defects)

    if rotation != 0:
        f = np.rot90(f, k=rotation)

    if flip_x:
        f = np.flip(f, 1)

    if flip_y:
        f = np.flip(f, 0)

    if power_spectrum:
        # Take the fourier transform of the image.
        f1 = fftpack.fft2(f)

        # Now shift the quadrants around so that low spatial frequencies are in
        # the center of the 2D fourier transformed image.
        f2 = fftpack.fftshift(f1)

        # Calculate a 2D power spectrum
        psd2D = np.abs(f2) ** 2

        return np.log10(psd2D)
    else:
        return f


def to_frames_gaussian(frames, lam, shape, super_res):
    # Calculate how many splits we need per frame, with a minimum of just 1 split
    n_splits = max(math.floor(multiprocessing.cpu_count() / len(frames)), 1)

    # Get (or precalculate) the gaussian distribution
    distribution = get_gauss_distribution(lam)

    # We can use a with statement to ensure threads are cleaned up promptly
    with multiprocessing.Pool() as pool:
        # Allow to pass the gaussian distribution
        func = partial(event_gaussian, distribution, shape, super_res)

        # Progress bar
        bar = tqdm.tqdm(total=n_splits * len(frames))

        def update(*a):
            bar.update(n_splits)

        results = list()

        for frame in frames:
            split = np.array_split(frame['d'], n_splits)

            # Assign jobs to workers
            results.append(pool.map_async(func, split, callback=update))

        result_frames = list()
        # Combine result into one frame again
        for result in results:
            result_frames.append(np.sum(result.get(), axis=0))

    return result_frames


def correct_defect_pixels(f, defects):
    masked = np.count_nonzero(defects)

    # Pick from a normal distribution, based on the image mean and stddev
    mean = np.mean(f)
    stddev = np.std(f)
    picks = np.random.normal(mean, stddev, masked)

    # Correct picks to never be negative
    picks[picks < 0] = 0

    # Convert to dtype of frame
    picks = picks.astype(f.dtype)

    # Place back
    np.place(f, defects, picks)

    return f


# Display some stats about the ToA timer
def timing_stats(hits, frames_idx, min_toa, max_toa, no_graph):
    global tpx3_tick
    toa = hits['ToA']

    print("ToA time start: %d" % toa[0])
    print("ToA time end: %d" % toa[-1])
    print("ToA time min: %d" % toa.min())
    print("ToA time max: %d" % toa.max())

    if toa.max() > 65535 - 100:
        ticks = 65535 - int(toa[0]) + int(toa[-1])
    else:
        ticks = int(toa[-1]) - int(toa[0])

    print("Exposure time (seconds): %.5f" % (ticks * tpx3_tick))

    if min_toa > 0 and max_toa > 0:
        print("Exposure time (marker) (seconds): %.5f" % ((max_toa - min_toa) * tpx3_tick))
    else:
        print("WARNING: No marker pixel data found for calculating exposure time. Using value above")
        min_toa = toa[0]
        max_toa = toa[-1]

    print("Frame start time (idx %d): %d" % (frames_idx[0]['start_idx'], toa[frames_idx[0]['start_idx']]))
    print("Frame end time (idx %d): %d" % (frames_idx[0]['end_idx'], toa[frames_idx[0]['end_idx']]))

    print("Event/Hit rate (MHit/s): %.1f" % ((len(hits) / 1000000.) / (ticks * tpx3_tick)))

    hit_rate, center_time, dtoa_time, beam_on_windows = calculate_local_hit_rate(hits, min_toa, max_toa)

    # Prepare plotting of timers
    if not no_graph:
        fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(9.6, 9.6))

        plot_timers(hits, frames_idx, ax0)
        plot_hit_rate(hit_rate, center_time, dtoa_time, beam_on_windows, frames_idx, ax1)
        plt.tight_layout()
        plt.show()


def calculate_local_hit_rate(hits, min_toa, max_toa):
    toa = hits['ToA']
    dtoa = toa - min_toa
    step = 1000000

    # Calculate hit rate frequency
    bins = np.arange(0, max_toa-min_toa, step)
    hist, bins = np.histogram(dtoa, bins=bins)
    center = (bins[:-1] + bins[1:]) / 2

    # Simple method to find beam-on time
    beam_on = (hist > np.max(hist) / 2)
    ind = list(np.where(np.diff(beam_on))[0])
    # Fix for beam_on at end and beginning of acquisition (conversion to Python list not pretty)
    if beam_on[0]:
        ind.insert(0, -1)
    if beam_on[-1]:
        ind.append(len(hist) - 1)
    beam_on_windows = np.array(ind).reshape(-1, 2)

    # Convert to rate
    hit_rate = hist / (step * tpx3_tick)

    # Convert to time
    center_time = center * tpx3_tick
    dtoa_time = dtoa * tpx3_tick

    # Beam on and off time
    for window in beam_on_windows:
        if window[0] == -1:
            start_time = 0
        else:
            start_time = center_time[window[0]]
        print("Beam on: %.5f" % start_time)
        print("Beam off: %.5f" % center_time[window[1]])

    return hit_rate, center_time, dtoa_time, beam_on_windows


def plot_hit_rate(local_hit_rate, center_time, dtoa_time, beam_on_windows, frames_idx, ax):
    # Plot
    ax.step(center_time, local_hit_rate, where='mid', color='red')

    # Beam on and off time
    for window in beam_on_windows:
        if window[0] == -1:
            start_time = 0
        else:
            start_time = center_time[window[0]]
        ax.axvspan(start_time, center_time[window[1]], alpha=0.2, color='green')

    # Frame start and end time
    for frame in frames_idx:
        ax.axvline(dtoa_time[frame['start_idx']])
        ax.axvline(dtoa_time[frame['end_idx']])

    # Format graph
    formatter0 = EngFormatter(unit='hit')
    ax.yaxis.set_major_formatter(formatter0)
    ax.set_title('Hit rate as function of time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Hit Rate (hits/s)')


# Plot ToA timer of entire run against hits
def plot_timers(hits, frames_idx, ax):
    index = np.arange(len(hits))

    for chip in range(0, 4):
        # Index frame to only the particular chip
        chip_events = hits[hits['chipId'] == chip]
        chip_index = index[hits['chipId'] == chip]

        # Get only every 1000nth hit
        toa = chip_events['ToA'][1::1000]
        toa_index = chip_index[1::1000]

        ax.scatter(toa_index, toa, label='Chip %d' % chip)

    ax.set_title('ToA as function of hit index (every 1000nth hit)')

    formatter0 = EngFormatter(unit='hit')
    ax.xaxis.set_major_formatter(formatter0)

    ax.set_xlabel('Hit index')
    ax.set_ylabel('ToA time')
    ax.legend()

    # Frame start and end time
    for frame_idx in frames_idx:
        ax.axvline(frame_idx['start_idx'])
        ax.axvline(frame_idx['end_idx'])


def print_cluster_stats(clusters, max_tot, max_size):
    print("WARNING: Limiting to first 1M clusters")
    limit = 1000000
    cluster_subset = clusters[0:limit, 0, :]

    # Only get cross events
    # cross_x = np.logical_and(events['x'] > 253, events['x'] < 260)
    # cross_y = np.logical_and(events['y'] > 253, events['y'] < 260)
    # cross = np.logical_or(cross_y, cross_x)
    # events_subset = events[cross]
    # tot = events_subset['sumToT']
    # size = events_subset['nHits']

    tot = np.sum(cluster_subset, axis=(1, 2))
    size = np.count_nonzero(cluster_subset, axis=(1, 2))

    # Figure
    try:
        fig = plt.figure(dpi=200)
        ax = fig.add_subplot(111)
    except _tkinter.TclError as e:
        print('Could not display cluster_stats plot. Error message was: %s' % str(e))
        return

    # Make 2d hist
    if max_tot is None:
        max_tot = np.percentile(tot, 99.99)

    if max_size is None:
        max_size = np.percentile(size, 99.999)

    cmap = copy.copy(plt.get_cmap('viridis'))
    cmap.set_under('w', 1)
    bins = [np.arange(0, max_tot, 25), np.arange(0, max_size, 1)]
    plt.hist2d(tot, size, cmap=cmap, vmin=0.000001, range=((0, max_tot), (0, max_size)), bins=bins,
               density=True)

    # Add box showing filter values
    ax.add_patch(
        patches.Rectangle(
            (clusters.attrs['cluster_min_sum_tot'], clusters.attrs['cluster_min_size']),  # (x,y)
            clusters.attrs['cluster_max_sum_tot'] - clusters.attrs['cluster_min_sum_tot'],  # width
            clusters.attrs['cluster_max_size'] - clusters.attrs['cluster_min_size'],  # height
            fill=False, edgecolor='red', linewidth=2, zorder=2
        )
    )

    # x-axis ticks
    xax = ax.get_xaxis()
    xax.set_major_locator(plt.MultipleLocator(50))
    xax.set_minor_locator(plt.MultipleLocator(25))
    xax.set_tick_params(colors='black', which='major')
    plt.xlabel('ToT Sum (A.U)')

    # y-axis ticks
    yax = ax.get_yaxis()
    yax.set_major_locator(plt.MultipleLocator(1))
    yax.set_tick_params(colors='black', which='major')

    ax.set_ylim(1)
    plt.ylabel('Cluster Size (pixels)')

    # Set grid
    plt.grid(b=True, which='both', zorder=1)

    # Color bar
    cbar = plt.colorbar()
    cbar.set_ticks([])
    cbar.set_label('Normalised occurrence')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
