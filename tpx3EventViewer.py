#!/usr/bin/env python
import _tkinter
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

VERSION = '2.0.0'
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

    data = f[source]

    if data.attrs['version'] != VERSION:
         print("WARNING: Version of data file does not match version of tpx3EventViewer (%s vs %s)" % (data.attrs['version'], VERSION))

    # Get z_source
    z_source = None
    if settings.hits_tot:
        z_source = 'ToT'
    elif settings.hits_toa:
        z_source = 'ToA'

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
        timing_stats(data, frames_idx)
        return 0

    gain = None
    if settings.gain:
        with mrcfile.open(settings.gain) as gain_file:
            gain = gain_file.data

    # Calculate all frames
    frames = list()
    for frame_idx in frames_idx:
        frames.append(to_frame(frame_idx['d'], z_source, settings.rotation, settings.flip_x, settings.flip_y,
                               settings.power_spectrum, shape, settings.super_res,
                               settings.normalize, gain))

    # Output
    if settings.t:
        if settings.f:
            filename = settings.f
        else:
            filename = os.path.splitext(settings.FILE)[0] + ".tif"

        save_tiff(frames, settings.uint32, settings.uint8, filename)

    if settings.m:
        if settings.f:
            filename = settings.f
        else:
            filename = os.path.splitext(settings.FILE)[0] + ".mrc"

        save_mrc(frames, filename)

    if not settings.n:
        show(frames, settings.animation)


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
    parser.add_argument("--timing_stats", action='store_true', help="Show timing stats")
    parser.add_argument("--tot_threshold", type=int, default=0, help="In hits show only hits above ToT threshold")
    parser.add_argument("--tot_limit", type=int, default=1023, help="In hits show only hits below ToT limit")
    parser.add_argument("--chip", type=int, default=None, help="Limit display to certain chip")
    parser.add_argument("--normalize", action='store_true', help="Normalize to the average (useful for showing ToT)")
    parser.add_argument("--exposure", type=float, default=0, help="Max exposure time in seconds (0: infinite)")
    parser.add_argument("--start", type=float, default=0, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=0, help="End time in seconds")
    parser.add_argument("--super_res", metavar='N', type=int, default=0,
                        help="Up scale the amount of pixels by N factor")
    parser.add_argument("--cluster_stats", action='store_true', help="Show cluster stats")
    parser.add_argument("--cluster_stats_tot", type=int, default=None, help="Override cluster_stats ToT limit")
    parser.add_argument("--cluster_stats_size", type=int, default=None, help="Override cluster_stats size limit")

    settings = parser.parse_args()

    return settings


def save_mrc(frames, filename):
    data = np.array(frames)

    data = np.flip(data, axes=(1,2))

    # Needs possible clipping to max uint16 values
    i16 = np.iinfo(np.uint16)
    if data.max() > i16.max:
        print("WARNING: Cannot fit in uint16. Clipping values to uint16 max.")
        np.clip(data, 0, i16.max, data)

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
    fig.canvas.set_window_title('tpx3EventViewer')
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.25, bottom=0.25)

    im = ax.imshow(frame, vmin=min5, vmax=max95)
    ax.format_coord = lambda x, y: '%3.2f, %3.2f, %10d' % (x, y, frame[int(y + 0.5), int(x + 0.5)])
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


def to_frame(frame, z_source, rotation, flip_x, flip_y, power_spectrum, shape, super_resolution, normalize, gain):
    x = frame['x']
    y = frame['y']

    if super_resolution > 0:
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

    if gain is not None:
        f = np.multiply(f, gain)

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


# Display some stats about the ToA timer
def timing_stats(hits, frames_idx):
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

    print("Frame start time (idx %d): %d" % (frames_idx[0]['start_idx'], toa[frames_idx[0]['start_idx']]))
    print("Frame end time (idx %d): %d" % (frames_idx[0]['end_idx'], toa[frames_idx[0]['end_idx']]))

    print("Event/Hit rate (MHit/s): %.1f" % ((len(hits) / 1000000.) / (ticks * tpx3_tick)))

    plot_timers(hits, frames_idx)


# Plot ToA timer of entire run against hits
def plot_timers(hits, frames_idx):
    fig, ax = plt.subplots()

    index = np.arange(len(hits))

    for chip in range(0, 4):
        # Index frame to only the particular chip
        chip_events = hits[hits['chipId'] == chip]
        chip_index = index[hits['chipId'] == chip]

        # Get only every 1000nth hit
        toa = chip_events['ToA'][1::1000]
        toa_index = chip_index[1::1000]

        plt.scatter(toa_index, toa, label='Chip %d' % chip)

    plt.title('ToA time (every 1000nth hit)')

    formatter0 = EngFormatter(unit='hit')
    ax.xaxis.set_major_formatter(formatter0)

    plt.xlabel('Hit index')
    plt.ylabel('Time ticks')
    plt.legend()

    # Frame start and end time
    for frame_idx in frames_idx:
        plt.axvline(frame_idx['start_idx'])
        plt.axvline(frame_idx['end_idx'])

    plt.show()


def print_cluster_stats(clusters, max_tot, max_size):
    print("WARNING: Limiting to first 1M clusters")
    limit = 1000000
    cluster_subset = clusters[0:limit, 0, :]

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

    # Colorbar
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
