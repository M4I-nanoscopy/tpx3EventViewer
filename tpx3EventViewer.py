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
    parser.add_argument("-n", action='store_true', help="Don't show interactive viewer")
    parser.add_argument("--hits", action='store_true', help="Use /hits instead of /events")
    # Super pixel option
    # Exposure time
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


# # This function yields frames from the main list, so is memory efficient
# def split(d, start_time, exposure):
#     frames = [list(), list(), list(), list(), list()]
#     last_frame = 0
#     frame_offset = 0
#
#     # TODO: Here we read only one chunk, this may be good place to multi thread??
#     events = d[0:d.chunks[0]]
#
#     for event in events:
#         # The SPIDR_TIME has about a 20s timer, it may reset mid run
#         if event[SPIDR_TIME] < start_time:
#             logger.debug("SPIDR_TIME reset to %d, from %d" % (event[SPIDR_TIME], start_time))
#             start_time = event[SPIDR_TIME]
#             frame_offset = last_frame + len(frames)
#
#         # Calculate current frame
#         frame = frame_offset + int((event[SPIDR_TIME] - start_time) / exposure)
#
#         if frame < last_frame:
#             logger.warn("Wrong order of events! %i - %d" % (frame, last_frame))
#             continue
#
#         # Yield previous frame if we are above buffer space of frames list
#         if frame >= last_frame + 5:
#             yield frames.pop(0)
#             # Add empty frame
#             frames.append(list())
#             # Increase counter
#             last_frame = last_frame + 1
#
#         frames[frame - last_frame].append(event)
#
#     # Yield remainder of frames
#     for frame in frames:
#         yield frame



if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
