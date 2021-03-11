import videos
import segments
import os

def create_face_videos(videos_path):
    basename = os.path.basename(videos_path)
    # fullpath = os.path.join(videos_path, video)
    df = videos.assign_face_ids(videos_path)

    if not os.path.exists("cropped_face_videos"):
        os.mkdir("cropped_face_videos")
    if not os.path.exists(os.path.join("cropped_face_videos", basename)):
        os.mkdir(os.path.join("cropped_face_videos", basename))

    # 3. create a video for each aligned face found
    aligned_videos_paths = []
    for id in df['id'].unique():
        # there may be breaks in frame continuity here (it's actually likely)
        frame_paths = df[df['id'] == id]['path']
        # the frames from the original video may be useful or necessary
        # frames = df[df['id'] == id]['frame']
        # if len(frame_paths) > 100: # todo: this doesnt always apply...
        if len(frame_paths) > 30:
            out_path = os.path.join("cropped_face_videos", basename) + "/" + str(int(id)) + ".mp4"
            # out_path = "videos/" + basename + "_aligned_" + str(int(id)) + ".mp4"
            videos.make_aligned_video(frame_paths, out_path)
            aligned_videos_paths.append(out_path)

# 0. downloading videos? change frame rate?
original_videos_directory = "original_videos"
face_images_basepath = "cropped_face_detections"
face_videos_basepath = "cropped_face_videos"
segment_videos_output_basepath = "segmented_videos"
if not os.path.isdir(segment_videos_output_basepath):
    os.mkdir(segment_videos_output_basepath)

for video in os.listdir(original_videos_directory):
    # 1. Detect and crop faces in a video by frame and output to a folder of images
    videoname_noext = os.path.splitext(video)[0]
    original_video_path = os.path.join(original_videos_directory, video)
    face_detections_path = os.path.join(face_images_basepath, videoname_noext)
    if not os.path.exists(face_detections_path):
        # todo: make check to see if the detections exist first
        videos.detect_faces(original_video_path)
        assert os.path.exists(face_detections_path)

    # 2. Make a video for each face in the video, and output to a folder of videos
    face_videos_path = os.path.join(face_videos_basepath, videoname_noext)
    if not os.path.exists(face_videos_path):
        create_face_videos(face_detections_path)
        # todo: move this function to "videos.py"
        assert os.path.exists(face_videos_path)

    # 3. Find expression segments for each cropped face video, output to a folder with a video per segment
    for face_video in os.listdir(face_videos_path):
        face_video_file = os.path.join(face_videos_path, face_video)
        au_df = segments.extract_features(face_video_file)
        au_df = segments.add_missing_frames(au_df, au_df['frame'].iloc[-1])
        # seg_df = segments.find_where_rolling_mean_deviates_from_threshold(au_df)
        seg_df = segments.find_segments_from_clusters(au_df, 0)
        # todo: face_id of 0 awkward, not general
        output_path = os.path.join(segment_videos_output_basepath, os.path.basename(face_videos_path))
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        for index, seg in seg_df.iterrows():
            output_video_file = output_path + "/" + str(index) + ".mp4"
            videos.save_sub_video(face_video_file, seg[0], seg[1], output_video_file)

exit(0)