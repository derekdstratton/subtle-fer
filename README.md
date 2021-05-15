# Subtle Facial Expression Recognition

* Derek Stratton
* Emily Hand
* Machine Perception Laboratory
* UNR Computer Science and Engineering

## Overview

Currently, this repo contains a bunch of modules to help aggregate
and label subtle FER data. `videos.py` has functions for downloading
and playing videos. `segments.py` uses OpenFace's AU detection to 
predict time segments where an expression occurs in a video. 
`labeling.py` is a GUI that takes a video and useful time segments, 
and lets the user label which subtle expression matches the clip.

## Requirements

* unix
* python 3
* some python packages, depending on the module

`segments.py`:
* installation of openface
  https://github.com/TadasBaltrusaitis/OpenFace/wiki/Unix-Installation
* video file

`labeling.py`:
* video file
* segments file

## Usage

`segments.py`

1. set the environment variable OPENFACE_PATH to the path where you installed
it. (By default it ends in `OpenFace/build/bin`)
   
2. Use python to run the script with the first argument being the basename of the video to find segments.
The video should be located in the `videos` subdirectory of this project. The second
argument should be a number for which method to run, currently supporting `1` or `2`.

3. It will output the segment in the `segment_labels` directory. OpenFace also
creates outputs in a `processed` directory. 

`labeling.py`

1. By default, it loads the `simple_test` video and segments. 

2. To load another video and segment, you can load the video from the `videos` folder.
The segments file should have the same base name, and be located in the `segment_labels` 
directory.
   
## making installation VM for labeling.py

1. `pyinstaller labeling.py` creates a folder called `dist` which packages everything
Copy and paste the pilot_segments folder into `dist/labeling`. Zip it and upload to the box to transport.
   
a. Note: MAKE SURE that labels.csv is NOT in the pilot_segments.

2. Download a VM image of Linux Lite from: https://www.osboxes.org/linux-lite/ and 
Virtualbox
   
3. Open Virtualbox, hit new. Put in the right options and import from hard disk,
selecting the .vdi image (it's debian based)
   
4. Download the `labeling.zip` file and put it on the desktop for easy access

5. Create an easy run script called `run_me.sh` that contains the following:
```
#!/bin/bash
./labeling
```

6. Run the command `chmod +x run_me.sh` to make it clickable from the desktop

7. Next, we need a GStreamer plugin to play the videos. https://gstreamer.freedesktop.org/documentation/installing/on-linux.html?gi-language=c

8. In Virtualbox, export applicance. And go to expert mode and change the file
extension to .ovf
   
9. 