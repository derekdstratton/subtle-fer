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
        subprocess.run([openface_build_path + "/FeatureExtraction", "-f", video_path])#, "-aus"])
    else:
        print("csv file already exists, loading into df")
    full_df = pd.read_csv(csv_from_video_path)
    # num_frames x num_AUs dataframe containing the AU intensity for each video frame
    # note: you can change the first slice here to look at a subset of video frames
    # note: i'm currently removing au 45, it gave me worse results
    return full_df.loc[:, 'AU01_r':'AU26_r']


# segment finding method 1
def find_where_rolling_mean_deviates_from_threshold(au_df):
    # hyperparameters:
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

    seg_df = segment_or_not_to_dataframe(any_au_deviates, play_segs=True)
    # output the resulting segments to file
    # seg_df.to_csv("segment_labels/" + video_basename.split('.')[0] + ".csv")
    return seg_df

# segment finding method 2
def find_segments_from_clusters():
    pass

# transforms a num_frames length bool series to a num_segments x 3 segments dataframe
# play_segs will play the segments it finds if true
def segment_or_not_to_dataframe(seg_or_not_series, play_segs=False):
    # https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-in-a-numpy-array
    def consecutive(data, stepsize=1):
        return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)

    list_of_segment_frames = consecutive(np.where(seg_or_not_series == True)[0])

    data = []
    if play_segs:
        import videos

    for seg in list_of_segment_frames:
        data.append((seg[0], seg[-1], 0))
        if play_segs:
            videos.play(video_name, seg[0], seg[-1])

    return pd.DataFrame(data, columns=['start_frame', 'end_frame', 'label'])

def plot_au_over_time(df, au_num, rolling_mean_window=20):
    # plt.plot(df['frame'], df['AU' + f'{au_num:02}' + '_r'])
    x = df['AU' + f'{au_num:02}' + '_r']
    df['interesting_segments' + 'AU' + f'{au_num:02}'] = plot_with_rolling(df['frame'], x, rolling_mean_window, au_num)


def plot_with_rolling(frame, au_over_time, rolling_mean_window, au_num):
    deviation = np.std(au_over_time)
    # smooths it out a little bit.
    rolling_mean = np.convolve(au_over_time, np.ones(rolling_mean_window) / rolling_mean_window, mode='same')
    df2['roll_' + 'AU' + f'{au_num:02}'] = rolling_mean
    # the full window gives something a bit clearer.
    # do I want to divide by 2?
    rolling_mean_with_full_window = np.convolve(au_over_time,
                                                np.ones(len(au_over_time) // 2) / (len(au_over_time) // 2), mode='same')

    # https://stackoverflow.com/questions/46964363/filtering-out-outliers-in-pandas-dataframe-with-rolling-median

    # plot the moving average
    plt.plot(frame, rolling_mean)
    plt.plot(frame, rolling_mean_with_full_window)
    plt.plot(frame, rolling_mean_with_full_window + deviation)
    plt.title("AU: " + str(au_num))
    plt.show()
    deviation = np.std(au_over_time)
    return rolling_mean > rolling_mean_with_full_window + 2 * deviation


def logic_on_au_features(df):
    # https://stackoverflow.com/questions/11285613/selecting-multiple-columns-in-a-pandas-dataframe
    # NOTE: i'm deciding to remove au45 since it's acting weird
    # (guess since i'm slicing this may also omit 26?)
    b = df.loc[:, 'interesting_segmentsAU01':'interesting_segmentsAU26']
    c = b.any(axis=1)
    df['interesting_segments'] = c
    plt.plot(df['frame'], c)
    plt.show()


# https://medium.com/@ansjin/dimensionality-reduction-using-pca-on-multivariate-timeseries-data-b5cc07238dc4
def pca_on_au_features(df, num_components=4):
    b = df.loc[:, 'roll_AU01':'roll_AU26']
    from sklearn.decomposition import PCA
    pca = PCA(n_components=num_components)
    return pca.fit_transform(np.array(b))

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


if __name__ == "__main__":
    # first argument should be the basename of the video
    video_basename = sys.argv[1]
    video_name = "videos/" + video_basename

    # get the AU dataframe
    au_df = extract_features(video_name)

    # use this method to find segments
    seg_df = find_where_rolling_mean_deviates_from_threshold(au_df)

    exit()


    df2 = df
    au_list = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]
    for au in au_list:
        # plot_au_over_time(df, au, 20)
        plot_au_over_time(df2, au, rolling_mean_window=50)
        # kinda like the bigger windows for not catching spikes
    logic_on_au_features(df2)
    interestings = df2['interesting_segments'][df2['interesting_segments'] == True].index
    # https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list

    interestings2 = df2['roll_AU06'][df2['roll_AU06'] > 0.2].index

    from itertools import groupby
    from operator import itemgetter

    segments = []
    # look for consecutive chains of true
    for k, g in groupby(enumerate(interestings2), lambda ix: ix[0] - ix[1]):
        segments.append(list(map(itemgetter(1), g)))

    seglens = [len(x) for x in segments]
    # maybe if len less than 50, it's just noise

    data = []
    os.environ['DISPLAY'] = "localhost:0.0"
    import videos
    for seg in segments:
        print(seg[0], seg[-1])
        data.append((seg[0], seg[-1], 0))
        # videos.play(video_name, seg[0], seg[-1])

    seg_df = pd.DataFrame(data, columns=['start_frame', 'end_frame', 'label'])
    # seg_df.to_csv("segment_labels/" + video_basename.split('.')[0] + ".csv")

    from sklearn import cluster
    c = pca_on_au_features(df, 8)
    km = cluster.KMeans() # default is 8 clusters
    km.fit(c)
    predikts = km.predict(c)
    plt.plot(predikts)
    plt.show()
    segments2 = []
    seg_groups = []
    # to find a single group, just use i = 3 or whichever num you want
    for i in range(0, 8):
        pred3 = np.where(predikts == i)
        # look for consecutive chains of true
        for k, g in groupby(enumerate(pred3[0]), lambda ix: ix[0] - ix[1]):
            segments2.append(list(map(itemgetter(1), g)))
            seg_groups.append(i)

    data2 = []
    # for seg in segments2:
    #     print(seg[0], seg[-1])
    #     data2.append((seg[0], seg[-1], 0))
        # videos.play(video_name, seg[0], seg[-1])
    seg2_sorted = sorted(segments2, key=lambda x: x[0])

    # this is a sequential list of each group corresponding to each seg
    seg2_groups_sorted = [x for _, x in sorted(zip(segments2, seg_groups), key=lambda x: x[0])]

    seglens2 = [len(x) for x in seg2_sorted]

    shortsegs_removed = []
    temp_seg = []
    # for getting the group without shortsegs
    groups_no_shortsegs = []
    i = 0
    # it's a naive approach grouping it with another randomly.
    # it should be going to whichever of the 2 side groups is
    # closer.
    for seg in seg2_sorted:
        temp_seg += seg
        if len(temp_seg) > 20:
            shortsegs_removed.append(temp_seg)
            temp_seg = []
            groups_no_shortsegs.append(seg2_groups_sorted[i])
        i += 1
    seglens_noshort = [len(x) for x in shortsegs_removed]

    i = 0
    for seg in shortsegs_removed:
        # this lets you filter to play segs based on group
        # if True:
        if groups_no_shortsegs[i] == 2:
            print(seg[0], seg[-1])
            data2.append((seg[0], seg[-1], 0))
            videos.play(video_name, seg[0], seg[-1])
        i += 1
    # maybe i should take super small segments, and just combine them
    # with closer ones automatically.
    # todo: consider trying agglomerative clustering, since
    # k = 8 is very arbitrary (granted so is k = 8 for PCA)