#!/usr/bin/env python

# Get rid of a harmless h5py FutureWarning. Can be removed with a new release of h5py
# https://github.com/h5py/h5py/issues/961
import _tkinter
import warnings

from matplotlib.lines import Line2D
from matplotlib.ticker import EngFormatter

warnings.filterwarnings('ignore', 'Conversion of the second argument of issubdtype from .*', )

import argparse
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
from matplotlib.widgets import Slider
from matplotlib import animation, patches
from PIL import Image
import os

VERSION = '0.5.1'
spidr_tick = 26.843 / 65536.
ticks_second = 1. / spidr_tick


def main():
    settings = parse_arguments()

    f = h5py.File(settings.FILE, 'r')

    if settings.cluster_stats:
        # TODO: Check for existence of these datasets, also this exit() here is not pretty
        print_cluster_stats(f['cluster_info'], f['cluster_stats'])
        exit(0)

    # Build frame
    if settings.hits:
        source = 'hits'
    else:
        source = 'events'

    data = f[source]

    # TODO: Enable this check later
    # if not data.attrs['version'] != VERSION:
    #     print "WARNING: Version of data file does not match version of tpx3EventViewer (%s vs %s)" % (data.attrs['version'], VERSION)

    z_source = None
    if settings.hits_tot:
        z_source = 'ToT'
    elif settings.hits_toa:
        z_source = 'cToA'
    elif settings.hits_spidr:
        z_source = 'TSPIDR'

    if 'shape' in data.attrs:
        shape = data.attrs['shape']
    else:
        # Backwards capability. This was the max size before implementing the shape attribute
        shape = 516

    # Determine frame indeces
    frames_idx = calculate_frames_idx(data, settings.exposure, settings.start, settings.end)

    if settings.spidr_stats:
        spidr_time_stats(data, frames_idx)

    # Calculate all frames
    frames = list()
    for frame_idx in frames_idx:
        frames.append(to_frame(frame_idx['d'], z_source, settings.rotation, settings.flip_x, settings.flip_y, shape))

    # Output
    if settings.t:
        if settings.f:
            filename = settings.f
        else:
            filename = os.path.splitext(settings.FILE)[0] + ".tif"

        save_tiff(frames, settings.uint32, settings.uint8, filename)

    if not settings.n:
        show(frames, settings.m)


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
    parser.add_argument("-f", metavar='FILE', help="File name for .tif file (default is .h5 file with .tif extension)")
    parser.add_argument("-n", action='store_true', help="Don't show interactive viewer")
    parser.add_argument("-m", action='store_true', help="Store as animated mp4 movie")
    parser.add_argument("-r", "--rotation", type=int, default=0, help="Rotate 90 degrees (1: clockwise, "
                                                                      "-1 anti-clockwise, 0: none). Default: 0")
    parser.add_argument("--flip_x", action='store_true', help="Flip image in X")
    parser.add_argument("--flip_y", action='store_true', help="Flip image in Y")

    parser.add_argument("--hits", action='store_true', help="Use hits (default in counting mode)")
    parser.add_argument("--hits_tot", action='store_true', help="Use hits in ToT mode")
    parser.add_argument("--hits_toa", action='store_true', help="Use hits in ToA mode")
    parser.add_argument("--hits_spidr", action='store_true', help="Use hits in SPIDR mode")
    parser.add_argument("--spidr_stats", action='store_true', help="Show SPIDR stats")
    parser.add_argument("--cluster_stats", action='store_true', help="Show cluster stats")
    parser.add_argument("--exposure", type=float, default=0, help="Max exposure time in seconds (0: infinite)")
    parser.add_argument("--start", type=float, default=0, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=0, help="End time in seconds")

    # Max number of frames

    settings = parser.parse_args()

    return settings


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
                print "WARNING: Cannot fit in uint8. Clipping values to uint16 max."
                np.clip(frame, 0, i8.max, frame)

            frame = frame.astype(dtype=np.uint8)

            im = Image.fromarray(frame)
        else:
            # Needs possible clipping to max uint16 values
            i16 = np.iinfo(np.uint16)
            if frame.max() > i16.max:
                print "WARNING: Cannot fit in uint16. Clipping values to uint16 max."
                np.clip(frame, 0, i16.max, frame)

            frame = frame.astype(dtype=np.uint16)

            im = Image.fromarray(frame)

        images.append(im)

    images[0].save(filename, save_all=True, append_images=images[1:])


def show(frames, animate):
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
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.25, bottom=0.25)

    im = ax.imshow(frame, vmin=min5, vmax=max95)
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

    axmin = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    axmax = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    smin = Slider(axmin, 'Min', min, max, valinit=min5)
    smax = Slider(axmax, 'Max', min, max, valinit=max95)
    smin.on_changed(update_clim)
    smax.on_changed(update_clim)

    if len(frames) > 1:
        axframe = fig.add_axes([0.25, 0.05, 0.65, 0.03])
        sframe = Slider(axframe, 'Frame', 0, len(frames) - 1, valinit=0, valfmt="%i")
        sframe.on_changed(update_frame)

    # ax.set(ylim=(375, 360), xlim=(433, 453), autoscale_on=False)
    # l = Line2D([430, 456], [460, 262], color='red')
    # ax.add_line(l)

    if animate:
        # Animate
        anim = animation.FuncAnimation(fig, animate_frame, frames=len(frames) - 1, interval=200)
        # Save
        anim.save('animation.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
    else:
        plt.show()


def calculate_frames_idx(data, exposure, start_time, end_time):
    frames = list()

    spidr = data['TSPIDR']

    if start_time > 0:
        start = spidr[0] + ticks_second * start_time
        start_idx = np.argmax(spidr > start)
    else:
        start = spidr[0]
        start_idx = 0

    if end_time > 0:
        end_final = spidr[0] + ticks_second * end_time
        end_final_idx = np.argmax(spidr > end_final)
        if end_final_idx == 0:
            end_final_idx = len(data) - 1
    else:
        end_final_idx = len(data) - 1

    end = start + ticks_second * exposure

    if exposure > 0:

        reset = False
        # Calculate all frames indeces
        while start_idx < end_final_idx:
            if reset:
                end_idx = np.argmax((spidr > end) & (spidr < spidr[0] - 10))

                # True end
                end = spidr[end_idx]

            elif end > 65535:
                remainder = start + ticks_second * exposure - 65535
                end_idx = np.argmax((spidr > remainder) & (spidr < spidr[0] - 10))

                end = spidr[end_idx]
                reset = True
            else:
                end_idx = np.argmax(spidr > end)

                # True end
                end = spidr[end_idx]

            # Correct for np.argmax returning 0 when going past the end
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

    return frames


def to_frame(frame, z_source, rotation, flip_x, flip_y, shape):
    # By casting to int we floor the result to the bottom left pixel it was found in
    rows = frame['y'].astype(dtype='uint16')
    cols = frame['x'].astype(dtype='uint16')

    if z_source is None:
        data = np.ones(len(frame))
    else:
        data = frame[z_source]

    d = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(shape, shape), dtype=np.uint32)
    f = d.todense()

    if rotation != 0:
        f = np.rot90(f, k=rotation)

    if flip_x:
        f = np.flip(f, 1)

    if flip_y:
        f = np.flip(f, 0)

    return f


# Display some stats about the SPIDR global time
def spidr_time_stats(hits, frames_idx):
    # Load all hits into memory
    hits = hits[()]

    spidr = hits['TSPIDR']

    tick = 26.843 / 65536.

    print("SPIDR time start: %d" % spidr[0])
    print("SPIDR time end: %d" % spidr[-1])
    print("SPIDR time min: %d" % spidr.min())
    print("SPIDR time max: %d" % spidr.max())

    if spidr.max() > 65535 - 100:
        ticks = 65535 - int(spidr[0]) + int(spidr[-1])
    else:
        ticks = int(spidr[-1]) - int(spidr[0])

    print("SPIDR exposure time (estimate): %.5f" % (ticks * tick))

    print("Frame start time (idx %d): %d" % (frames_idx[0]['start_idx'], spidr[frames_idx[0]['start_idx']]))
    print("Frame end time (idx %d): %d" % (frames_idx[0]['end_idx'], spidr[frames_idx[0]['end_idx']]))

    print("Event/Hit rate (MHit/s): %.1f" % ((len(hits) / 1000000.) / (ticks * tick)))

    plot_timers(hits, frames_idx)


# Plot SPIDR time of entire run
def plot_timers(hits, frames_idx):
    fig, ax = plt.subplots()

    index = np.arange(len(hits))

    for chip in range(0, 4):
        # Index frame to only the particular chip
        chip_events = hits[[hits['chipId'] == chip]]
        chip_index = index[[hits['chipId'] == chip]]

        # Get only every 1000nth hit
        spidr = chip_events['TSPIDR'][1::1000]
        spidr_index = chip_index[1::1000]

        plt.scatter(spidr_index, spidr, label='Chip %d' % chip)

    plt.title('SPIDR time (every 1000nth hit)')

    formatter0 = EngFormatter(unit='hit')
    ax.xaxis.set_major_formatter(formatter0)

    plt.xlabel('Hit index')
    plt.ylabel('SPIDR time ticks')
    plt.legend()

    # Frame start and end time
    for frame_idx in frames_idx:
        plt.axvline(frame_idx['start_idx'])
        plt.axvline(frame_idx['end_idx'])

    plt.show()


def print_cluster_stats(cluster_info, cluster_stats):
    stats = cluster_stats[()]

    removed = len(cluster_stats) - len(cluster_info)
    removed_percentage = float(removed / float(len(cluster_info) + removed)) * 100

    print("Removed %d clusters and single hits (%d percent)" % (removed, removed_percentage))

    # Figure
    try:
        fig, ax = plt.subplots()
    except _tkinter.TclError as e:
        print('Could not display cluster_stats plot. Error message was: %s' % str(e))
        return

    # Make 2d hist
    cmap = plt.get_cmap('viridis')
    cmap.set_under('w', 1)
    bins = [np.arange(0, 700, 25), np.arange(0, 16, 1)]
    plt.hist2d(stats[:, 1], stats[:, 0], cmap=cmap, vmin=1, range=((0, 700), (0, 16)), bins=bins)

    # Add box showing filter values
    ax.add_patch(
        patches.Rectangle(
            (cluster_stats.attrs['cluster_min_sum_tot'], cluster_stats.attrs['cluster_min_size']),  # (x,y)
            cluster_stats.attrs['cluster_max_sum_tot'] - cluster_stats.attrs['cluster_min_sum_tot'],  # width
            cluster_stats.attrs['cluster_max_size'] - cluster_stats.attrs['cluster_min_size'],  # height
            fill=False, edgecolor='red', linewidth=2
        )
    )

    ax.set_xticks(bins[0])
    ax.set_yticks(bins[1])
    ax.set_ylim(1)
    plt.tick_params(colors='black', )
    plt.grid(b=True, which='both')
    plt.ylabel('Cluster Size (pixels)')
    plt.xlabel('Cluster Total ToT (A.U)')
    plt.colorbar()

    fig, ax = plt.subplots()
    plt.xlabel('Cluster Total ToT (A.U)')
    plt.ylabel('Normalised occurrence')
    plt.grid()
    plt.hist(cluster_stats[:, 1], range=(0, 700), bins=699)

    plt.show()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
