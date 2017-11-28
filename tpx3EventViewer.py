#!/usr/bin/env python

import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
from matplotlib.widgets import Slider
from PIL import Image

# TODO: Is this the most elegant way?
E_CHIP = 0
E_X = 1
E_Y = 2
E_TIME = 3


def main():
    f = h5py.File(sys.argv[1], 'r')

    frame = build(f['events'][()])

    im = Image.fromarray(frame, 'L')
    #im.save('test.tif')
    show(frame)


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
        chip_events = frame[[frame[:, E_CHIP] == chip]]

        rows = chip_events[:, E_X]
        cols = chip_events[:, E_Y]

        hist = np.histogram2d(rows, cols, bins=(xedges, yedges))
        chip_frame = hist[0]
        #chip_frame = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(256, 256))

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
