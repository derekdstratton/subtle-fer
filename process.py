import videos
import segments
import os

basename = "simple_test"
# basename = "When_Among_Us_Arguments_Go_TOO_FAR_-_Pokimane_&_Valkyrae_Rage"
# basename = "Psychic_Cringe_Fails_1_-_HILARIOUS_REACTION"

# 0. download? change frame rate?

# 1. use face detection algorithm to create image folder of faces
# (it's possible it detect bad faces, or miss faces)
# NOTE: this needs the trained SVM
# todo: make check to see if it exists first
videos.detect_faces("videos/" + basename + ".mp4")

aligned_videos_paths = []
# if already contains these, just load them
for video in os.listdir("videos"):
    if "aligned" in video and basename in video:
        aligned_videos_paths.append(video)

# 2. use face verification algorithm to iterate over images and assign ids
# (it's possible to mismatch)
# todo: be able to save and load from this step and next step
if len(aligned_videos_paths) == 0:
    df = videos.assign_face_ids(basename)

    # 3. create a video for each aligned face found
    aligned_videos_paths = []
    for id in df['id'].unique():
        # there may be breaks in frame continuity here (it's actually likely)
        frame_paths = df[df['id'] == id]['path']
        # the frames from the original video may be useful or necessary
        # frames = df[df['id'] == id]['frame']
        if len(frame_paths) > 100:
            out_path = "videos/" + basename + "_aligned_" + str(int(id)) + ".mp4"
            videos.make_aligned_video(frame_paths, out_path)
            aligned_videos_paths.append(out_path)

# 4. find segments for each aligned video
# you need openface installed, and the OPENFACE_PATH defined
seg_dfs = {}
for aligned_vid in aligned_videos_paths:
    try:
        au_df = segments.extract_features(aligned_vid)
        au_df = segments.add_missing_frames(au_df, au_df['frame'].iloc[-1])
        seg_df = segments.find_segments_from_clusters(au_df, 0) # todo: face_id of 0 awkward
        seg_df = segments.delete_segments_by_length(seg_df, smallest_len=30, largest_len=300)
        seg_df.to_csv("segment_labels/" + os.path.basename(aligned_vid).split('.')[0] + ".csv", index=False)
        seg_dfs[aligned_vid] = seg_df
    except:
        # lol
        continue