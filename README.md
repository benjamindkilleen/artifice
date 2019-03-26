# Artifice

Object detection for scientific images.

## Installation

We provide instructions for using `artifice` UChicago's `midway2` cluster. After
cloning this repository, add its directory to the `PYTHONPATH`:
```
git clone https://github.com/bendkill/artifice.git
cd artifice
ARTIFICE_ROOT=`pwd`
export PYTHONPATH=$PYTHONPATH:$ARTIFICE_ROOT
```

To run `artifice`, open a compute node. We show here with GPU node: 
```
sinteractive --partition=gpu2 --gres=gpu:1 --mem=32000
```

After it opens, load the required modules and activate the python
environment. Note: you may have to `module unload python` first.
```
module load Anaconda3/2018.12
source activate tf-gpu-1.12.0
module load povray/3.7
module load ffmpeg
```

(We use `POV-Ray` for data generation and `ffmpeg` for visualization).

Check that `artifice` loads correctly by running `python artifice.py -h`

## Demo

Artifice's default command-line options are configured for this demo. All
commands should be run from the `ARTIFICE_ROOT` directory.

This demonstration runs object detection using an artificial dataset with two
bouncing spheres, under noisy conditions. By default, only 10 video frames are
labeled for training. Detections are run on 1000 withheld frames.

1. The `test_utils` directory contains `experiment.py` for creating an
   artificial experiment. Run ```python scripts/coupled_spheres.py``` to create
   the test dataset, viualized
   [here](https://github.com/bendkill/artifice/blob/master/docs/coupled_spheres.mp4).
   (This will take about an hour, avoid if possible.) The dataset consists of
   `.png` images and a `.npy` labels file.
2. Convert this data to the expected `.tfrecord` form with
   ```python artifice.py convert```
3. Run one training epoch with ```python artifice.py train```. (Epoch
   checkpoints are saved in `models/coupled_spheres/hourglass`
4. Run object detection with ```python artifice.py detect``` This creates a
   `detections.npy` file with object detections in `models/coupled_spheres`.
5. Run ```python artifice.py visualize``` to create and save detection
   visualizations in `models/coupled_spheres`. Examples are included with this
   distribution.
