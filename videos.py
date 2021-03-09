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

def detect_faces(video_path):
    if not os.path.exists("detections"):
        os.mkdir("detections")

    import face_detection
    detector = face_detection.build_detector(
        "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
    cap = cv2.VideoCapture(video_path)

    base_name = os.path.basename(video_path).split(".")[0]
    if not os.path.exists("detections/" + base_name + "/faces"):
        os.mkdir("detections/" + base_name)
        os.mkdir("detections/" + base_name + "/faces")

    # todo: make sure this folder exists.
    #
    # Create a VideoCapture object and read from input file
    # Check if camera opened successfully
    # https://subscription.packtpub.com/book/application_development/9781788474443/1/ch01lvl1sec24/jumping-between-frames-in-video-files
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if (cap.isOpened() == False):
        print("Error opening video  file")

        # Read until video is completed
    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()
        # to stop a video, while frame < STOPPING_NUM
        if ret == True:

            # Display the resulting frame
            # cv2.imshow('Frame', frame)
            detections = detector.detect(frame)
            frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)

            for i, face_bounds in enumerate(detections):
                if frame_num == 1.0:
                    width = int(face_bounds[3]) - int(face_bounds[1])
                    height = int(face_bounds[2]) - int(face_bounds[0])
                    x_lower = int(face_bounds[1]) - (width // 2)
                    x_upper = int(face_bounds[3]) + (width // 2)
                    y_lower = int(face_bounds[0]) - (height // 2)
                    y_upper = int(face_bounds[2]) + (height // 2)

                # larger window (naive, still jittery)
                # width = int(face_bounds[3]) - int(face_bounds[1])
                # height = int(face_bounds[2]) - int(face_bounds[0])
                # cropped = frame[int(face_bounds[1]) - (width // 2):int(face_bounds[3]) + (width // 2),
                #           int(face_bounds[0]) - (height // 2):int(face_bounds[2]) + (height // 2)]

                # possibly better: find a general centroid so it doesnt jitter
                cropped = frame[x_lower:x_upper,y_lower:y_upper]
                # original close crop
                # cropped = frame[int(face_bounds[1]):int(face_bounds[3]),
                #           int(face_bounds[0]):int(face_bounds[2])]
                # pass
                if cropped.size > 0 and face_bounds[4] > 0.9:# and (cropped.size / frame.size) > 0.02:
                    cv2.imwrite("detections/" + base_name + "/faces/" +
                                str(frame_num) + "_" + str(i) + ".jpg", cropped)
            print(cap.get(cv2.CAP_PROP_POS_FRAMES))
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
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

import torch.nn as nn
import torch
import torchvision.models as models


def assign_face_ids(video_basename):

    import torch
    import facenet_pytorch

    model = facenet_pytorch.InceptionResnetV1(pretrained='vggface2').eval()

    import torchvision
    import torchvision.transforms as transforms
    import numpy as np
    # tt = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize(256),
    #     torchvision.transforms.ToTensor()]
    # )
    tt = transforms.Compose([
        transforms.Resize((160,160)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = torchvision.datasets.ImageFolder("detections/" + video_basename, transform=tt)
    dataloader = torch.utils.data.DataLoader(dataset)#,sampler=np.arange(1000))
    anchor_embeddings = {}  # int key, tensor value

    found = False
    ids = np.zeros(len(dataloader))
    counter = 0
    import tqdm
    import matplotlib.pyplot as plt

    arr = np.zeros((len(dataloader), 512)) # (512 is the embedding length

    from joblib import load
    svm = load("compare.joblib")

    for idx, data in tqdm.tqdm(enumerate(dataloader)):
        embedding = model(data[0][0].unsqueeze(0).float())
        if idx < 10:
            plt.imshow(data[0][0].numpy().transpose(1,2,0))
            plt.title(str(idx))
            plt.show()
        arr[idx] = embedding.detach().numpy()

        ####

        similarities = {}
        for id, anchor_embed in anchor_embeddings.items():
            # dist = torch.norm(embedding - anchor_embed)
            # comparison = siamese(data[0], anchor_embed)
            # comparison = torch.norm(embedding - anchor_embed)
            ee = embedding - anchor_embed
            pred = svm.predict(ee.detach().numpy())
            # similarities[comparison] = id
            if pred == 1:
                found = True
                ids[idx] = id
                break
            # if comparison.item() > 0.5:
            # # if dist < 0.3:
            #     found = True
            #     ids[idx] = id
        # print(similarities.keys())
        if len(similarities) > 0:
            # closest = max(similarities.keys())
            # if closest > 0.95:
            #     found = True
            #     ids[idx] = similarities[closest]
            closest = min(similarities.keys())
            if closest < 1.0:
                found = True
                ids[idx] = similarities[closest]
        if not found:
            ids[idx] = counter
            anchor_embeddings[counter] = embedding
            # anchor_embeddings[counter] = data[0]
            print(counter)
            counter += 1

        found = False

        ####

    import pandas as pd
    df = pd.DataFrame({'path': [x[0] for x in dataset.samples],
                       'frame': [int(os.path.basename(x[0]).split('.')[0]) for x in dataset.samples],
                       'id': ids})
    # df = pd.DataFrame({'path': [x[0] for x in dataset.samples[:1000]],
    #                    'frame': [int(os.path.basename(x[0]).split('.')[0]) for x in dataset.samples[:1000]],
    #                    'id': ids})
    df = df.sort_values('frame')
    df = df.astype({'id': 'int16'})
    return df

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

def change_frame_rate(video_path, frame_rate=15):
    # https://stackoverflow.com/questions/53767770/python-video-decrease-fps
    # https://stackoverflow.com/questions/29317262/opencv-video-saving-in-python
    out = cv2.VideoWriter("test.mp4")
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.cv.CV_CAP_PROP_FPS, frame_rate)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            # frame = cv2.flip(frame, 0)

            # write the flipped frame
            out.write(frame)

            # cv2.imshow('frame', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def train_svm_on_lfw():
    from sklearn.datasets import fetch_lfw_pairs
    stuff = fetch_lfw_pairs()
    import torchvision
    import torchvision.transforms as transforms
    import numpy as np
    # tt = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize(256),
    #     torchvision.transforms.ToTensor()]
    # )
    tt = transforms.Compose([
        transforms.Resize((160,160)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    arr = np.zeros((len(stuff.pairs), 512))
    import facenet_pytorch
    from PIL import Image
    import tqdm
    model = facenet_pytorch.InceptionResnetV1(pretrained='vggface2').eval()
    for idx, val in tqdm.tqdm(enumerate(stuff.pairs)):
        em1 = model(tt(Image.fromarray(val[0]).convert("RGB")).view(1,3,160,160)) #...something like that
        em2 = model(tt(Image.fromarray(val[1]).convert("RGB")).view(1,3,160,160))
        ee = em1 - em2
        arr[idx] = ee.detach().numpy()

    from sklearn.svm import SVC
    fit = SVC().fit(arr, stuff.target)
    from joblib import dump
    dump(fit, "compare.joblib")

    return arr, stuff.target

def make_aligned_video(frame_paths, out_path):
    # (assuming all samples are the same size)
    sample = cv2.imread(frame_paths[0])
    out_dim = (sample.shape[1], sample.shape[0])
    #

    out = cv2.VideoWriter(out_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          30,
                          out_dim
                          )
    for p in frame_paths:
        a = cv2.imread(p)
        a = cv2.resize(a, out_dim)
        out.write(a)
    out.release()


def pls(files_path):
    # files_path = "A1_AU1_TrailNo_2"
    # li = []
    # for video in sorted(os.listdir(files_path)):
    #     li.append(os.path.join(files_path, video))
    # make_aligned_video(li, "videos/pls.mp4")

    import os
    base_paths = ["SN001", "SN003", "SN004", "SN007", "SN009", "SN010",
                  "SN013", "SN025", "SN027"]
    for base_path in base_paths:
        for path in os.listdir(base_path):
            li = []
            full_path = os.path.join(base_path, path)
            if os.path.isdir(full_path):
                for video in sorted(os.listdir(full_path)):
                    if ".jpg" in video:
                        li.append(os.path.join(full_path, video))
                print(li)
                videos.make_aligned_video(li, "videos/" + base_path + path + ".mp4")
