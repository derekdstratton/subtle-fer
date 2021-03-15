import os
import sys

import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import numpy as np

# todo: it would be nice to have options to toggle plots on/off, toggle playing videos on/off


# takes a video file, and returns a dataframe with all the AUs
# uses OpenFace to find the AUs. it will skip processing if it finds it's already been processed
# https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units
# todo: make a -o arg
def extract_features(video_path):
    fixed_path = video_path.replace("/", "_")
    fixed_path = os.path.splitext(fixed_path)[0]
    fixed_path = os.path.join("processed", fixed_path)
    csv_file = os.path.join(fixed_path, os.path.splitext(os.path.basename(video_path))[0] + ".csv")
    if not os.path.isfile(csv_file):
        openface_build_path = os.environ['OPENFACE_PATH']

        # for a single person
        # os.system(openface_build_path + "/FeatureExtraction")
        # os.system(openface_build_path + "/FeatureExtraction")

        # https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments
        subprocess.run([openface_build_path + "/FeatureExtraction", "-f", video_path, "-out_dir", fixed_path])
        # subprocess.run([openface_build_path + "/FaceLandmarkVidMulti", "-f", video_path])
    else:
        print("csv file already exists, loading into df")
    full_df = pd.read_csv(csv_file)
    # num_frames x num_AUs dataframe containing the AU intensity for each video frame
    # note: you can change the first slice here to look at a subset of video frames
    # note: i'm currently removing au 45 (blinking), it gave me worse results.
    col_indices = list(range(list(full_df.columns).index('AU01_r'), list(full_df.columns).index('AU26_r')+1))
    col_indices.append(0)
    col_indices.append(1)
    return full_df.iloc[:, col_indices]
    # return dict(tuple(full_df.iloc[:, col_indices].groupby('face_id')))
    # return full_df.loc[:, 'AU01_r':'AU26_r']


# segment finding method 1
def find_where_rolling_mean_deviates_from_threshold(au_df, manual_labels=None):
    # hyperparameters/adjustable things:
    # rolling mean window and method (currently 50)
    # threshold (currently median + 1 std)
    # number of accepted au's (currently if any 1 deviates)

    # num_frames x num_AUs dataframe, using a rolling average to attempt to smooth points
    smoothed_au_df = au_df.rolling(10, min_periods=1, center=True).mean()

    # so stupidly simple but it kinda works LOL
    norm = np.linalg.norm(smoothed_au_df[smoothed_au_df.columns.drop('face_id')], axis=1)
    threshold = np.median(norm) + np.std(norm)
    bools = norm > threshold
    plt.plot(norm)
    plt.show()

    # plots of the AUs are useful
    # for col in smoothed_au_df.columns.drop('face_id'):
    #     plot_au(smoothed_au_df[col], col, "Smoothed AU Plot: " + col, manual_labels)

    # num_frames x num_AUs boolean dataframe, where it's true if the smoothed value is > 1 std away from median
    # au_deviants_df = pd.DataFrame()
    # for col in smoothed_au_df.columns.drop('face_id'):
    #     threshold = smoothed_au_df[col].median() + smoothed_au_df[col].std()
    #     au_deviants_df[col] = smoothed_au_df[col] > threshold
    #
    # # num_frames boolean series if any of the AUs deviate
    # any_au_deviates = au_deviants_df.any(axis=1)

    # plots where segments are based on if any au deviates from median
    plot_segment_or_not(bools, 0, manual_labels=manual_labels)

    return segment_or_not_to_dataframe(au_df.index, bools)


# segment finding method 2
def find_segments_from_clusters(au_df, face_id, manual_labels=None):
    # hyperparameters/adjustable things:
    # number of pca components (currently 8), and using this to reduce dimensionality
    # the clustering method (currently agglomerative)
    # distance_threshold for agglomerative or num_clusters for k means
    # method for determining useful groups (currently any group that isn't the main)

    # np.linalg.norm(au_df_nona[au_df_nona.columns.drop('face_id')], axis=1)

    # smoothing seems like a good idea
    au_df = au_df.rolling(50, min_periods=1, center=True).mean()

    # plotting the smoothed au's
    # for col in au_df.columns.drop('face_id'):
    #     plot_au(au_df[col], col, "AU Plot: " + col, manual_seg_or_not)

    # to use PCA/KMeans, we need to drop nans (missing values)
    au_df_nona = au_df.dropna()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=4)
    # num_frames x num_components array
    aus_transformed = pca.fit_transform(au_df_nona[au_df_nona.columns.drop('face_id')])
    # how many dimensions (components) is reasonable to cluster?

    from sklearn import cluster
    # k means: i think this is just worse than agglomerative
    kmeans = cluster.KMeans(4)  # default is 8 clusters
    kmeans.fit(aus_transformed)
    # num_frames length array with each value being from 0 to num_groups - 1
    group_for_each_frame = kmeans.predict(aus_transformed)

    # agglomerative: the distance_threshold seems very sensitive
    # agg = cluster.AgglomerativeClustering(n_clusters=None, distance_threshold=20)
    # agg.fit(aus_transformed)
    # # num_frames length array with each value being from 0 to num_groups - 1
    # group_for_each_frame = agg.labels_

    # import hdbscan
    # clusterer = hdbscan.HDBSCAN(metric='manhattan', prediction_data=True)
    # clusterer.fit(aus_transformed)

    # todo: this also needs reindexed to handle mia values
    # plot_groups(group_for_each_frame, face_id)
    # the idea of this choice is that the closest cluster center to all 0s AUs is probably the boring one
    base = pca.transform(np.zeros((1,16)))
    choice = np.argmin(np.linalg.norm(base[0] - kmeans.cluster_centers_, axis=1))
    not_in_most_common_group = group_for_each_frame != choice
    # one problem with clusters is knowing which cluster(s) are interesting or not
    # i'm just saying if it's not in the most common group, it's interesting.
    # most_common_group = np.bincount(group_for_each_frame).argmax()
    # not_in_most_common_group = group_for_each_frame != most_common_group


    aa1 = pd.DataFrame(not_in_most_common_group)
    aa2 = aa1.set_index(au_df_nona.index)
    aa3 = aa2.reindex(au_df.index)

    # plots where segments are based on if any au deviates from median
    plot_segment_or_not(aa3, face_id, manual_labels=manual_labels)

    return segment_or_not_to_dataframe(au_df.index, not_in_most_common_group)


# transforms a num_frames length bool series to a num_segments x 3 segments dataframe
# play_segs will play the segments it finds if true
def segment_or_not_to_dataframe(frame, seg_or_not_series):
    # https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-in-a-numpy-array
    def consecutive(arr, stepsize=1):
        return np.split(arr, np.where(np.diff(arr) != stepsize)[0] + 1)

    list_of_segment_frames = consecutive(np.where(seg_or_not_series)[0])

    data = []

    for seg in list_of_segment_frames:
        startF = int(frame[seg[0]])
        endF = int(frame[seg[-1]])
        data.append((startF, endF, 0))

    return pd.DataFrame(data, columns=['start_frame', 'end_frame', 'label'])


def dataframe_to_seg_or_not(seg_df, total_frames):
    seg_or_not = np.zeros((total_frames,))
    for index, seg in seg_df.iterrows():
        np.put(seg_or_not, list(range(seg['start_frame'], seg['end_frame'])), 1)
    return seg_or_not


def summarize_segments(seg_df):
    binwidth = 5
    seg_lens = (seg_df['end_frame'] - seg_df['start_frame']).astype('int32')
    print(seg_lens.describe())
    plt.hist(seg_lens, bins=range(min(seg_lens) - (min(seg_lens) % 5), max(seg_lens) + (max(seg_lens) % 5) + binwidth, binwidth))
    plt.xlabel("Segment Length")
    plt.ylabel("Number of Occurences")
    plt.title("Histogram of Segment Lengths: Bins of Size 5")
    plt.show()
    return seg_lens


def delete_segments_by_length(seg_df, smallest_len=20, largest_len=200):
    seg_lens = (seg_df['end_frame'] - seg_df['start_frame']).astype('int32')
    return seg_df[(seg_lens >= smallest_len) & (seg_lens <= largest_len)]


def plot_segment_or_not(seg_or_not_series, face_id, manual_labels=None):
    plt.plot(seg_or_not_series, label="predicted segments")
    if manual_labels is not None:
        # shift by 1.5 so they don't overlap visually
        plt.plot(manual_labels * 1.5, label="manual segments", linestyle="dashed")
    plt.xlabel("frame")
    plt.ylabel("segment or not")
    plt.title("Detected Segments for each Frame, face_id=" + str(face_id))
    plt.legend()
    plt.show()


def plot_au(au_series, au_name, title, manual_labels=None):
    plt.plot(au_series.index, au_series, label=au_name)
    plt.plot(np.full(len(au_series.index), au_series.median()), label="median")
    plt.plot(np.full(len(au_series.index), au_series.median() + au_series.std()), label="median plus 1 std")
    if manual_labels is not None:
        plt.plot(au_series.index, manual_labels, label="manual segments", linestyle='dashed')
    plt.xlabel("frame")
    plt.ylabel(au_name)
    plt.title(title)
    # plt.legend()
    plt.show()


def plot_groups(groups_series, face_id):
    plt.plot(groups_series)
    plt.xlabel("frame")
    plt.ylabel("group")
    plt.title("Detected Groups for each Frame, face_id: " + str(face_id))
    plt.show()


# they are represented as nan
# this also makes the index the frames between 1 and the last frame
def add_missing_frames(au_df, last_frame):
    # https://stackoverflow.com/questions/25909984/missing-data-insert-rows-in-pandas-and-fill-with-nan
    new_index = pd.Index(np.arange(1, last_frame))
    return au_df.set_index("frame").reindex(new_index)


def basic_information(au_df):
    print("Number of total frames: " + str(au_df.index[-1]))
    print("Number of missing frames: " + str(au_df['face_id'].isna().sum()))
    print("Missing values: " + str(au_df.index[au_df['face_id'].isna()]))


def feature_picker(au_df, desired_aus):
    desired_aus.append("face_id")
    return au_df[desired_aus]


def play_segments(seg_df):
    import videos
    for index, seg in seg_df.iterrows():
        videos.play(video_name, seg['start_frame'], seg['end_frame'])


if __name__ == "__main__":
    # first argument should be the basename of the video
    video_basename = sys.argv[1]
    video_name = "videos/" + video_basename

    # second argument should be which method to run
    method_to_run = sys.argv[2]

    # get the AU dataframe
    all_au_dfs = extract_features(video_name)
    # last frame is the final from every seen face
    last_frame = max([df['frame'].iloc[-1] for k, df in all_au_dfs.items()])
    all_seg_dfs = {}
    for face_id, au_df in all_au_dfs.items():

        # make sure that missing frames are accounted for
        au_df = add_missing_frames(au_df, last_frame)

        basic_information(au_df)

        # au_df = feature_picker(au_df, ['AU06_r', 'AU20_r', 'AU01_r', 'AU02_r'])

        # if manual segments, we might want to see that
        manual_seg_or_not = None
        if os.path.exists("segment_labels/" + video_basename.split('.')[0] + "_manual" + str(face_id) + ".csv"):
            manual_seg_df = pd.read_csv("segment_labels/" + video_basename.split('.')[0] + "_manual" + str(face_id) + ".csv")
            manual_seg_or_not = dataframe_to_seg_or_not(manual_seg_df, au_df.index[-1])

        # plotting the au's
        # for col in au_df.columns.drop('face_id'):
        #     plot_au(au_df[col], col, "AU Plot: " + col + ", Face ID: " + str(face_id), manual_seg_or_not)

        seg_df = None

        if int(method_to_run) == 1:
            # method 1 for segments
            seg_df = find_where_rolling_mean_deviates_from_threshold(au_df, manual_labels=manual_seg_or_not)
        elif int(method_to_run) == 2:
            # method 2 for segments
            seg_df = find_segments_from_clusters(au_df, face_id, manual_labels=manual_seg_or_not)
        else:
            print("Invalid method to run: " + method_to_run)

        if seg_df is not None:
            seg_lens = summarize_segments(seg_df)

            # edit segments to not have too short or too long
            seg_df = delete_segments_by_length(seg_df, smallest_len=30, largest_len=300)

            # round trip (for testing)
            thing = dataframe_to_seg_or_not(seg_df, au_df.index[-1])

            # output the resulting segments to file
            seg_df.to_csv("segment_labels/" + video_basename.split('.')[0] + str(face_id) + ".csv", index=False)

            all_seg_dfs[face_id] = seg_df