import videos
import segments
import os

# 0. downloading videos? change frame rate?
# original_videos_directory = "original_videos"
original_videos_directory = "EMMA_subset"
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
        videos.create_face_videos(face_detections_path)
        assert os.path.exists(face_videos_path)

    # 3. Find expression segments for each cropped face video, output to a folder with a video per segment
    for face_video in os.listdir(face_videos_path):
        face_video_file = os.path.join(face_videos_path, face_video)
        au_df = segments.extract_features(face_video_file)
        au_df = segments.add_missing_frames(au_df, au_df['frame'].iloc[-1])
        seg_df = segments.find_where_rolling_mean_deviates_from_threshold(au_df)
        # seg_df = segments.find_segments_from_clusters(au_df, 0)
        seg_df = segments.delete_segments_by_length(seg_df, smallest_len=20, largest_len=300)
        output_path = os.path.join(segment_videos_output_basepath, os.path.basename(face_videos_path))
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        for index, seg in seg_df.iterrows():
            output_video_file = output_path + "/" + str(index) + ".mp4"
            videos.save_sub_video(face_video_file, seg[0], seg[1], output_video_file)