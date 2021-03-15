import os
import cv2

# downloads a YouTube video
def try_download(link):
    from pytube import YouTube
    import demoji
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
        name = ys.title.replace(" ", "_")
        name = demoji.replace(name, "")
        ys.download(output_path="videos", filename=name)
        print("Download completed!!")
    else:
        print("File already exists")
    return ys.default_filename.replace(" ", "_")


# outputs all images from a video to a file
# video_path: path to a video file
# return: path to the image folder of outputs
def detect_faces(video_path: str) -> str:
    import face_detection

    # pathing
    faces_output_basepath = "cropped_face_detections"
    if not os.path.exists(faces_output_basepath):
        os.mkdir(faces_output_basepath)
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    faces_video_path = os.path.join(faces_output_basepath, video_basename)
    if not os.path.exists(faces_video_path):
        os.mkdir(faces_video_path)
    faces_video_path_inner = os.path.join(faces_video_path, "faces")
    if not os.path.exists(faces_video_path_inner):
        os.mkdir(faces_video_path_inner)

    # load face detector and the video
    detector = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not cap.isOpened():
        print("Error opening video file")
    found = False
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
            # the list of detections, equal to amount of faces found in a frame
            detections = detector.detect(frame)
            for i, face_bounds in enumerate(detections):
                if not found:
                    width = int(face_bounds[3]) - int(face_bounds[1])
                    height = int(face_bounds[2]) - int(face_bounds[0])
                    x_lower = int(face_bounds[1]) - (width // 2)
                    x_upper = int(face_bounds[3]) + (width // 2)
                    y_lower = int(face_bounds[0]) - (height // 2)
                    y_upper = int(face_bounds[2]) + (height // 2)
                    found = True

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
                    # cv2.imwrite("detections/" + video_basename + "/faces/" +
                    #             str(frame_num) + "_" + str(i) + ".jpg", cropped)
                    cv2.imwrite(faces_video_path_inner + "/" + str(frame_num) + "_" + str(i) + ".jpg", cropped)
            print(cap.get(cv2.CAP_PROP_POS_FRAMES))
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    return faces_video_path

# takes face groups, detects faces that are the same, and makes them into their own video
# face_detections_path: a path to a folder that contains images of a single face
# returns the path to the folder containing all the face videos
def create_face_videos(face_detections_path: str) -> str:
    import torch
    import facenet_pytorch
    import torchvision
    import torchvision.transforms as transforms
    import numpy as np
    import tqdm
    from joblib import load
    import pandas as pd

    # todo: make sure this exists
    svm = load("compare.joblib")

    model = facenet_pytorch.InceptionResnetV1(pretrained='vggface2').eval()

    tt = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
    ])
    # dataset = torchvision.datasets.ImageFolder("detections/" + video_basename, transform=tt)
    dataset = torchvision.datasets.ImageFolder(face_detections_path, transform=tt)
    dataloader = torch.utils.data.DataLoader(dataset)  # ,sampler=np.arange(1000))
    anchor_embeddings = {}  # int key, tensor value

    found = False
    ids = np.zeros(len(dataloader))
    counter = 0
    arr = np.zeros((len(dataloader), 512))  # (512 is the embedding length

    for idx, data in tqdm.tqdm(enumerate(dataloader)):
        embedding = model(data[0][0].unsqueeze(0).float())
        arr[idx] = embedding.detach().numpy()
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

    df = pd.DataFrame({'path': [x[0] for x in dataset.samples],
                       'frame': [int(os.path.basename(x[0]).split('.')[0]) for x in dataset.samples],
                       'id': ids})
    df = df.sort_values('frame')
    df = df.astype({'id': 'int16'})

    basename = os.path.basename(face_detections_path)
    if not os.path.exists("cropped_face_videos"):
        os.mkdir("cropped_face_videos")
    face_videos_output_path = os.path.join("cropped_face_videos", basename)
    if not os.path.exists(face_videos_output_path):
        os.mkdir(face_videos_output_path)

    for id in df['id'].unique():
        # there may be breaks in frame continuity here (it's actually likely)
        frame_paths = df[df['id'] == id]['path']
        # the frames from the original video may be useful or necessary
        # frames = df[df['id'] == id]['frame']
        # if len(frame_paths) > 100: # todo: this doesnt always apply...
        if len(frame_paths) > 30:
            out_path = face_videos_output_path + "/" + str(int(id)) + ".mp4"
            image_sequence_to_video(frame_paths, out_path)

    return face_videos_output_path


# takes a video and plays it
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


# takes a video, start, and end frame, and saves that bit as its own video
def save_sub_video(video_path, start_frame, end_frame, out_path):
    cap = cv2.VideoCapture(video_path)
    # Create a VideoCapture object and read from input file
    # Check if camera opened successfully
    # https://subscription.packtpub.com/book/application_development/9781788474443/1/ch01lvl1sec24/jumping-between-frames-in-video-files

    out_dim = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #

    out = cv2.VideoWriter(out_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          30,
                          out_dim
                          )

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
            # cv2.imshow('Frame', frame)
            out.write(frame)
            print(cap.get(cv2.CAP_PROP_POS_FRAMES))
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            if end_frame is not None and cap.get(cv2.CAP_PROP_POS_FRAMES) >= end_frame:
                break

        # Break the loop
        else:
            break

    out.release()
    # When everything done, release
    # the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    return cap

# takes a video, and creates a new video with specified frame rate
def change_frame_rate(video_path, out_path, frame_rate=15):
    # https://stackoverflow.com/questions/53767770/python-video-decrease-fps
    # https://stackoverflow.com/questions/29317262/opencv-video-saving-in-python
    out = cv2.VideoWriter(out_path)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.cv.CV_CAP_PROP_FPS, frame_rate)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            out.write(frame)
        else:
            break
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# creates a face verification model
def train_svm_on_lfw():
    from sklearn.datasets import fetch_lfw_pairs
    import torchvision
    import torchvision.transforms as transforms
    import numpy as np
    import facenet_pytorch
    from PIL import Image
    import tqdm
    from sklearn.svm import SVC
    from joblib import dump

    stuff = fetch_lfw_pairs()
    tt = transforms.Compose([
        transforms.Resize((160,160)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    arr = np.zeros((len(stuff.pairs), 512))

    model = facenet_pytorch.InceptionResnetV1(pretrained='vggface2').eval()
    for idx, val in tqdm.tqdm(enumerate(stuff.pairs)):
        em1 = model(tt(Image.fromarray(val[0]).convert("RGB")).view(1,3,160,160)) #...something like that
        em2 = model(tt(Image.fromarray(val[1]).convert("RGB")).view(1,3,160,160))
        ee = em1 - em2
        arr[idx] = ee.detach().numpy()

    fit = SVC().fit(arr, stuff.target)

    dump(fit, "compare.joblib")

    return arr, stuff.target

# takes a series of image paths and concatenates them into a video
def image_sequence_to_video(frame_paths, out_path):
    # (assuming all samples are the same size)
    sample = cv2.imread(frame_paths.iloc[0])
    out_dim = (sample.shape[1], sample.shape[0])

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