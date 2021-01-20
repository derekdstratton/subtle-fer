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
`labeling.py` is a  GUI that takes a video and useful time segments, 
and lets the user label which subtle expression matches the clip.

## Requirements

* unix
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
   
2. Run the script with the argument being the path of the video to find segments

3. It will output the segment in the `segment_labels` directory. OpenFace also
creates outputs in a `processed` directory. 

`labeling.py`

1. By default, it loads the `simple_test` video and segments. 

2. To load another video and segment, you can load the video from the `videos` folder.
The segments file should have the same base name, and be located in the `segment_labels` 
   directory.