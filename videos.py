import os
from pytube import YouTube
import cv2

video_path = "videos/simple_test.mp4"


# os.environ['DISPLAY'] = "localhost:0.0"
# print(os.environ['DISPLAY'])

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
