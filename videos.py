from pytube import YouTube
import cv2
import numpy

# getting ffmpeg to work on wsl, building ope
# ncv for openface: https://github.com/justadudewhohacks/opencv4nodejs/issues/274
import os
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import numpy as np

# the interest-times are VERY approximate eyeballing
video_path = "videos/simple_test.mp4"
au_list = [1,2,4,5,6,7,9,10,12,14,15,17,20,23,25,26,45]
# os.environ['DISPLAY'] = "localhost:0.0"
# print(os.environ['DISPLAY'])

# todo: make an if to check if it's already processed, with a separate function

# ask for the link from user, and download it
# returns mp4 filename
# https://towardsdatascience.com/build-a-youtube-downloader-with-python-8ef2e6915d97
def try_download(link):
    yt = YouTube(link)

    # Showing details
    print("Title: ", yt.title)
    print("Number of views: ", yt.views)
    print("Length of video: ", yt.length)
    print("Rating of video: ", yt.rating)
    # Getting the highest resolution possible
    ys = yt.streams.get_highest_resolution()

    if not os.path.isfile(ys.default_filename):
        # Starting download
        print("Downloading...")
        # ys.download()
        import demoji
        name = ys.title.replace(" ", "_")
        name = demoji.replace(name, "")
        ys.download(output_path="videos", filename=name)
        print("Download completed!!")
    else:
        print("File already exists")
    return ys.default_filename.replace(" ", "_")

# https://www.geeksforgeeks.org/python-play-a-video-using-opencv/
def play(video_path, start_frame=0, end_frame=None):
    cap = cv2.VideoCapture(video_path)
    # Create a VideoCapture object and read from input file
    #
    # Check if camera opened successfully
    # https://subscription.packtpub.com/book/application_development/9781788474443/1/ch01lvl1sec24/jumping-between-frames-in-video-files
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if (cap.isOpened() == False):
        print("Error opening video  file")

        # Read until video is completed
    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()
        # to stop a video, while frame < STOPPING_NUM
        if ret == True:

            # Display the resulting frame
            cv2.imshow('Frame', frame)
            print(cap.get(cv2.CAP_PROP_POS_FRAMES))
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            if end_frame is not None and cap.get(cv2.CAP_PROP_POS_FRAMES) >= end_frame:
                break

        # Break the loop
        else:
            break

    # When everything done, release
    # the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    return cap

# maybe we need some sort of way to rank the AUs.
# i should probably go thru a messy youtube video (with talking) and find some parts to extract...
# and then see if i can get it to find them.
# (talking is the most awkward part with AU detection)
# maybe make AU thresholds different for those parts

# note, not all vids have caption data
# 1 simple way to handle it is just to not consider voice parts
# this gets tricky with multiple people
# caption = yt.captions.get_by_language_code('en')
# temp = caption.generate_srt_captions()
# you can get the frames by multiplying ys.fps by the time in seconds from srt
# segment lengths are also really varied. they can be way too
# long or way too short


from segments import *

if __name__ == "__main__":
    video_path2 = "https://www.youtube.com/watch?v=b0GxmVCd-FE"
    # video name needs renamed something that doesn't have spaces
    video_name = try_download(video_path2)
    df = extract_features(video_name)
    # z = play(video_path)

    # c = pca_on_au_features(df)

    # plot_au_over_time(df, 25, 1041)

    # todo: make a for loop to graph all the au's
    df2 = df.iloc[4000:6000,]
    for au in au_list:
        # plot_au_over_time(df, au, 20)
        plot_au_over_time(df2, au, rolling_mean_window=50)
        # kinda like the bigger windows for not catching spikes
    logic_on_au_features(df2)
    interestings = df2['interesting_segments'][df2['interesting_segments']==True].index
    # https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list

    from itertools import groupby
    from operator import itemgetter
    segments = []
    for k, g in groupby(enumerate(interestings), lambda ix : ix[0] - ix[1]):
        segments.append(list(map(itemgetter(1), g)))

    seglens = [len(x) for x in segments]
    # maybe if len less than 50, it's just noise
    for seg in segments:
        print(seg[0], seg[-1])
        play("cringe.mp4", seg[0], seg[-1])


    # replacing larger signal values to get a better idea of the main curve
