#!/usr/bin/env python

# Get rid of a harmless h5py FutureWarning. Can be removed with a new release of h5py
# https://github.com/h5py/h5py/issues/961
import warnings

from matplotlib.ticker import EngFormatter

warnings.filterwarnings('ignore', 'Conversion of the second argument of issubdtype from .*', )

import argparse
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
from matplotlib.widgets import Slider
from PIL import Image
import os

VERSION = '0.5.1'
spidr_tick = 26.843 / 65536.
ticks_second = 1. / spidr_tick

def main():
    settings = parse_arguments()

    f = h5py.File(settings.FILE, 'r')

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

    if settings.exposure > 0:
        spidr = data['TSPIDR']
        start = spidr[0]
        start_idx = 0

        end = start + ticks_second * settings.exposure

        if end > 65535:
            remainder = start + ticks_second * settings.exposure - 65535
            end_idx = np.argmax((spidr > remainder) & (spidr < spidr[0] - 10))
        else:
            end_idx = np.argmax(spidr > end)

        d = data[start_idx:end_idx]
    else:
        start_idx = 0
        end_idx = len(data)
        d = data[()]

    if settings.spidr_stats:
        spidr_time_stats(data, start_idx, end_idx)

    if 'shape' in data.attrs:
        shape = data.attrs['shape']
    else:
        # Backwards capability. This was the max size before implementing the shape attribute
        shape = 516

    frame = to_frame(d, z_source, settings.rotation, settings.flip_x, settings.flip_y, shape)

    # Output
    if settings.t:

        if settings.uint32:
            # Can store directly to image
            im = Image.fromarray(frame)
        else:
            # Needs possible clipping to max uint16 values
            i16 = np.iinfo(np.uint16)
            if frame.max() > i16.max:
                print "WARNING: Cannot fit in uint16. Clipping values to uint16 max."
                np.clip(frame, 0, i16.max, frame)

            frame = frame.astype(dtype=np.uint16)

            im = Image.fromarray(frame)

        if settings.f:
            filename = settings.f
        else:
            filename = os.path.splitext(settings.FILE)[0] + ".tif"

        im.save(filename)

    if not settings.n:
        show(frame)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,  # printed with -h/--help
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('FILE', help="Input .h5 file")
    parser.add_argument("-t", action='store_true', help="Save 16 bit uint .tif file")
    parser.add_argument("--uint32", action='store_true', help="Store uint32 tif (not supported by all readers!)")
    parser.add_argument("-f", metavar='FILE', help="File name for .tif file (default is .h5 file with .tif extension)")
    parser.add_argument("-n", action='store_true', help="Don't show interactive viewer")
    parser.add_argument("-r", "--rotation", type=int, default=0, help="Rotate 90 degrees (1: clockwise, "
                                                                      "-1 anti-clockwise, 0: none). Default: 0")
    parser.add_argument("--flip_x", action='store_true', help="Flip image in X")
    parser.add_argument("--flip_y", action='store_true', help="Flip image in Y")

    parser.add_argument("--hits", action='store_true', help="Use hits (default in counting mode)")
    parser.add_argument("--hits_tot", action='store_true', help="Use hits in ToT mode")
    parser.add_argument("--hits_toa", action='store_true', help="Use hits in ToA mode")
    parser.add_argument("--hits_spidr", action='store_true', help="Use hits in SPIDR mode")
    parser.add_argument("--exposure", type=float, default=0, help="Max exposure time in seconds (0: infinite)")
    parser.add_argument("--spidr_stats", action='store_true', help="Show SPIDR stats")

    # Max number of frames

    settings = parser.parse_args()

    return settings


def show(frame):
    # TODO: Make this cleaner

    # Calculate threshold values
    min5 = np.percentile(frame, 5)
    min = np.min(frame)
    max95 = np.percentile(frame, 95)
    max = np.max(frame)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.25, bottom=0.25)

    im1 = ax.imshow(frame, vmin=min5, vmax=max95)
    fig.colorbar(im1)

    axmin = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    axmax = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    smin = Slider(axmin, 'Min', min, max, valinit=min5)
    smax = Slider(axmax, 'Max', min, max, valinit=max95)

    def update(val):
        im1.set_clim([smin.val, smax.val])
        fig.canvas.draw()

    smin.on_changed(update)
    smax.on_changed(update)

    plt.show()


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
def spidr_time_stats(hits, start_idx, end_idx):
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

    print("Frame start time (idx %d): %d" % (start_idx, spidr[start_idx]))
    print("Frame end time (idx %d): %d" % (end_idx, spidr[end_idx]))

    print("Event/Hit rate (MHit/s): %.1f" % ((len(hits) / 1000000) / (ticks * tick)))

    plot_timers(hits, start_idx, end_idx)


# Plot SPIDR time of entire run
def plot_timers(hits, start_idx, end_idx):
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
    plt.axvline(start_idx)
    plt.axvline(end_idx)

    plt.show()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
