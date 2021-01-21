import os
import sys

import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import numpy as np


# takes a video file, and returns a dataframe with all the AUs
# uses OpenFace to find the AUs. it will skip processing if it finds it's already been processed
# https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units
def extract_features(video_path):
    base_path = os.path.basename(video_path)
    csv_from_video_path = "processed/" + base_path.split('.')[0] + ".csv"
    if not os.path.isfile(csv_from_video_path):
        openface_build_path = os.environ['OPENFACE_PATH']

        # for a single person
        os.system(openface_build_path + "/FeatureExtraction")

        # https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments
        subprocess.run([openface_build_path + "/FeatureExtraction", "-f", video_path])  # , "-aus"])
    else:
        print("csv file already exists, loading into df")
    full_df = pd.read_csv(csv_from_video_path)
    # num_frames x num_AUs dataframe containing the AU intensity for each video frame
    # note: you can change the first slice here to look at a subset of video frames
    # note: i'm currently removing au 45, it gave me worse results
    return full_df.loc[:, 'AU01_r':'AU26_r']


# segment finding method 1
def find_where_rolling_mean_deviates_from_threshold(au_df):
    # hyperparameters/adjustable things:
    # rolling mean window and method (currently 50)
    # threshold (currently median + 1 std)
    # number of accepted au's (currently if any 1 deviates)

    # num_frames x num_AUs dataframe, using a rolling average to attempt to smooth points
    smoothed_au_df = au_df.rolling(50).mean()

    # plots of the AUs are useful
    for col in smoothed_au_df.columns:
        plot_au(smoothed_au_df[col], col)

    # num_frames x num_AUs boolean dataframe, where it's true if the smoothed value is > 1 std away from median
    au_deviants_df = pd.DataFrame()
    for col in smoothed_au_df.columns:
        threshold = smoothed_au_df[col].median() + smoothed_au_df[col].std()
        au_deviants_df[col] = smoothed_au_df[col] > threshold

    # num_frames boolean series if any of the AUs deviate
    any_au_deviates = au_deviants_df.any(axis=1)

    # plots where segments are based on if any au deviates from median
    plot_segment_or_not(any_au_deviates)

    return segment_or_not_to_dataframe(any_au_deviates, play_segs=True)


# segment finding method 2
def find_segments_from_clusters(au_df):
    # hyperparameters/adjustable things:
    # number of pca components (currently 8), and using this to reduce dimensionality
    # the clustering method (currently agglomerative
    # distance_threshold for agglomerative or num_clusters for k means
    # method for determining useful groups (currently any group that isn't the main)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=8)
    # num_frames x num_components array
    aus_transformed = pca.fit_transform(au_df)
    # how many dimensions (components) is reasonable to cluster?

    from sklearn import cluster
    # k means: i think this is just worse than agglomerative
    # kmeans = cluster.KMeans()  # default is 8 clusters
    # kmeans.fit(aus_transformed)
    # # num_frames length array with each value being from 0 to num_groups - 1
    # group_for_each_frame = kmeans.predict(aus_transformed)
    # plot_groups(group_for_each_frame)

    # agglomerative: the distance_threshold seems very sensitive
    agg = cluster.AgglomerativeClustering(n_clusters=None, distance_threshold=20)
    agg.fit(aus_transformed)
    # num_frames length array with each value being from 0 to num_groups - 1
    group_for_each_frame = agg.labels_
    plot_groups(group_for_each_frame)

    # one problem with clusters is knowing which cluster(s) are interesting or not
    # i'm just saying if it's not in the most common group, it's interesting.
    most_common_group = np.bincount(group_for_each_frame).argmax()
    not_in_most_common_group = group_for_each_frame != most_common_group

    # plots where segments are based on if any au deviates from median
    plot_segment_or_not(not_in_most_common_group)

    return segment_or_not_to_dataframe(not_in_most_common_group, play_segs=True)


# transforms a num_frames length bool series to a num_segments x 3 segments dataframe
# play_segs will play the segments it finds if true
def segment_or_not_to_dataframe(seg_or_not_series, play_segs=False):
    # https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-in-a-numpy-array
    def consecutive(arr, stepsize=1):
        return np.split(arr, np.where(np.diff(arr) != stepsize)[0] + 1)

    list_of_segment_frames = consecutive(np.where(seg_or_not_series)[0])

    data = []
    if play_segs:
        import videos

    for seg in list_of_segment_frames:
        data.append((seg[0], seg[-1], 0))
        if play_segs:
            videos.play(video_name, seg[0], seg[-1])

    return pd.DataFrame(data, columns=['start_frame', 'end_frame', 'label'])


def plot_segment_or_not(seg_or_not_series):
    plt.plot(seg_or_not_series)
    plt.xlabel("frame")
    plt.ylabel("segment or not")
    plt.show()


def plot_au(au_series, au_name):
    plt.plot(au_series, label=au_name)
    plt.plot(np.full(len(au_series), au_series.median()), label="median")
    plt.plot(np.full(len(au_series), au_series.median() + au_series.std()), label="median plus 1 std")
    plt.xlabel("frame")
    plt.ylabel(au_name)
    plt.legend()
    plt.show()


def plot_groups(groups_series):
    plt.plot(groups_series)
    plt.xlabel("frame")
    plt.ylabel("group")
    plt.show()


if __name__ == "__main__":
    # first argument should be the basename of the video
    video_basename = sys.argv[1]
    video_name = "videos/" + video_basename

    # second argument should be which method to run
    method_to_run = sys.argv[2]

    # get the AU dataframe
    au_df = extract_features(video_name)

    if int(method_to_run) == 1:
        # method 1 for segments
        seg_df = find_where_rolling_mean_deviates_from_threshold(au_df)
    elif int(method_to_run) == 2:
        # method 2 for segments
        seg_df = find_segments_from_clusters(au_df)

    # output the resulting segments to file
    # seg_df.to_csv("segment_labels/" + video_basename.split('.')[0] + ".csv")
