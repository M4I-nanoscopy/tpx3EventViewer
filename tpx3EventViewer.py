#!/usr/bin/env python
import argparse
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from PIL import Image
import os


def main():
    settings = parse_arguments()

    f = h5py.File(settings.FILE, 'r')

    # Build frame
    if settings.hits:
        source = 'hits'
    else:
        source = 'events'

    frame = build(f[source][()])

    # Output
    if settings.t:
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
    parser.add_argument("-t", action='store_true', help="Save 8-bit .tif file")
    parser.add_argument("-f", metavar='FILE', help="File name for .tif file (default is .h5 file with .tif extension)")
    parser.add_argument("-n", action='store_true', help="Don't show matplotlib frame")
    parser.add_argument("--hits", action='store_true', help="Use /hits instead of /events")

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


def build(events):
    # TODO: Handle exposure time and multiple frames
    # frames = list()
    # if exposure is not None:
    #     start_time = d[0][SPIDR_TIME]
    #     frames = split(d, start_time, exposure)
    #
    # for events in frames:
    #

    frame = events_to_frame(events)
    return frame


# This function converts the event data to a 256 by 256 matrix and places the chips into a full frame
# Also deals with the chip positions
def events_to_frame(frame):
    img = np.zeros(shape=(512, 512), dtype=np.uint16)

    xedges = np.arange(0, 257, 1)
    yedges = np.arange(0, 257, 1)

    for chip in range(0, 4):
        # Index frame to only the particular chip
        chip_events = frame[[frame['chipId'] == chip]]

        rows = chip_events['x']
        cols = chip_events['y']

        hist = np.histogram2d(rows, cols, bins=(xedges, yedges))
        chip_frame = hist[0]
        # chip_frame = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(256, 256))

        # if chip == 0:
        #     img[256:512, 256:512] = np.rot90(chip_frame.todense(), k=1)
        # if chip == 1:
        #     img[0:256, 256:512] = np.rot90(chip_frame.todense(), k=-1)
        # if chip == 2:
        #     img[0:256, 0:256] = np.rot90(chip_frame.todense(), k=-1)
        # if chip == 3:
        #     img[256:512, 0:256] = np.rot90(chip_frame.todense(), k=1)
        #
        # TODO: Chips position need to be configurable
        if chip == 0:
            img[256:512, 256:512] = np.rot90(chip_frame, k=1)
        if chip == 1:
            img[0:256, 256:512] = np.rot90(chip_frame, k=-1)
        if chip == 2:
            img[0:256, 0:256] = np.rot90(chip_frame, k=-1)
        if chip == 3:
            img[256:512, 0:256] = np.rot90(chip_frame, k=1)

        # if chip == 0:
        #     img[512:1024, 512:1024] = np.rot90(chip_frame, k=1)
        # if chip == 1:
        #     img[0:512, 512:1024] = np.rot90(chip_frame, k=-1)
        # if chip == 2:
        #     img[0:512, 0:512] = np.rot90(chip_frame, k=-1)
        # if chip == 3:
        #     img[512:1024, 0:512] = np.rot90(chip_frame, k=1)

    return img


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
