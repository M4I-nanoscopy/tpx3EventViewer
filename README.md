## Getting ready

Download

```
git clone https://github.com/M4I-nanoscopy/tpx3EventViewer.git
cd tpx3EventViewer
```

Recommended way is to use a Python virtualenv. But this is optional.

```
virtualenv tpx3
source tpx3/bin/activate
```

Install Python dependencies

```
pip install -r requirements.txt
```

## Running

```
$ ./tpx3EventViewer.py --help
usage: tpx3EventViewer.py 

positional arguments:
  FILE                  Input .h5 file

optional arguments:
  -h, --help            show this help message and exit
  -t                    Store uint16 .tif file
  --uint32              Store uint32 tif (not supported by all readers!)
  --uint8               Store uint8 tif (supported by almost all readers)
  -f FILE               File name for .tif file (default is .h5 file with .tif
                        extension)
  -n                    Don't show interactive viewer
  -m                    Store as animated mp4 movie
  -r ROTATION, --rotation ROTATION
                        Rotate 90 degrees (1: clockwise, -1 anti-clockwise, 0:
                        none). Default: 0
  --power_spectrum      Show power spectrum
  --flip_x              Flip image in X
  --flip_y              Flip image in Y
  --hits                Use hits (default in counting mode)
  --hits_tot            Use hits in ToT mode
  --hits_toa            Use hits in ToA mode
  --hits_ctoa           Use hits in cToA mode
  --hits_ftoa           Use hits in fToA mode
  --hits_spidr          Use hits in SPIDR mode
  --spidr_stats         Show SPIDR stats
  --tot_threshold TOT_THRESHOLD
                        In hits show only hits above ToT threshold
  --tot_limit TOT_LIMIT
                        In hits show only hits below ToT limit
  --chip CHIP           Limit display to certain chip
  --events_sumtot       Show event in sumToT
  --normalize           Normalize ToT, ToA, fTOA or events-sumToT to number of
                        hits/events (the average)
  --exposure EXPOSURE   Max exposure time in seconds (0: infinite)
  --start START         Start time in seconds
  --end END             End time in seconds
  --super_res N         Up scale the amount of pixels by N factor
  --cluster_stats       Show cluster stats
  --cluster_stats_tot CLUSTER_STATS_TOT
                        Override cluster_stats ToT limit
  --cluster_stats_size CLUSTER_STATS_SIZE
                        Override cluster_stats size limit
```

## Copyright

(c) Maastricht University

## License

MIT license

## Authors

Paul van Schayck (p.vanschayck@maastrichtuniversity.nl)