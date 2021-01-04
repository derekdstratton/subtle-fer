import os
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import numpy as np


def extract_features(video_path):
    base_path = os.path.basename(video_path)
    csv_from_video_path = "processed/" + base_path.split('.')[0] + ".csv"
    if not os.path.isfile(csv_from_video_path):
        openface_build_path = "/mnt/c/Users/derek/PycharmProjects/OpenFace/build/bin"

        # for a single person
        os.system(openface_build_path + "/FeatureExtraction")

        # https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments
        subprocess.run([openface_build_path + "/FeatureExtraction", "-f", video_path])
    else:
        print("csv file already exists, loading into df")
    return pd.read_csv(csv_from_video_path)


def plot_au_over_time(df, au_num, rolling_mean_window=20):
    plt.plot(df['frame'], df['AU' + f'{au_num:02}' + '_r'])
    x = df['AU' + f'{au_num:02}' + '_r']
    df['interesting_segments' + 'AU' + f'{au_num:02}'] = plot_with_rolling(df['frame'], x, rolling_mean_window)


def plot_with_rolling(frame, au_over_time, rolling_mean_window):
    deviation = np.std(au_over_time)
    # smooths it out a little bit.
    rolling_mean = np.convolve(au_over_time, np.ones(rolling_mean_window) / rolling_mean_window, mode='same')

    # the full window gives something a bit clearer.
    # do I want to divide by 2?
    rolling_mean_with_full_window = np.convolve(au_over_time,
                                                np.ones(len(au_over_time) // 2) / (len(au_over_time) // 2), mode='same')

    # https://stackoverflow.com/questions/46964363/filtering-out-outliers-in-pandas-dataframe-with-rolling-median

    # plot the moving average
    plt.plot(frame, rolling_mean)
    plt.plot(frame, rolling_mean_with_full_window)
    plt.plot(frame, rolling_mean_with_full_window + deviation)
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
def pca_on_au_features(df):
    b = df.loc[:, 'AU01_r':'AU26_r']
    from sklearn.decomposition import PCA
    pca = PCA(n_components=4)
    return pca.fit_transform(np.array(b))


if __name__ == "__main__":
    video_basename = "simple_test.mp4"
    video_name = "videos/" + video_basename
    df = extract_features(video_name)
    # z = play(video_path)

    # c = pca_on_au_features(df)

    # plot_au_over_time(df, 25, 1041)

    # df2 = df.iloc[4000:6000,]
    au_list = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]
    df2 = df
    for au in au_list:
        # plot_au_over_time(df, au, 20)
        plot_au_over_time(df2, au, rolling_mean_window=50)
        # kinda like the bigger windows for not catching spikes
    logic_on_au_features(df2)
    interestings = df2['interesting_segments'][df2['interesting_segments'] == True].index
    # https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list

    from itertools import groupby
    from operator import itemgetter

    segments = []
    # look for consecutive chains of true
    for k, g in groupby(enumerate(interestings), lambda ix: ix[0] - ix[1]):
        segments.append(list(map(itemgetter(1), g)))

    seglens = [len(x) for x in segments]
    # maybe if len less than 50, it's just noise

    data = []
    for seg in segments:
        print(seg[0], seg[-1])
        data.append((seg[0], seg[-1], 0))
        # play(video_name, seg[0], seg[-1])

    seg_df = pd.DataFrame(data, columns=['start_frame', 'end_frame', 'label'])
    seg_df.to_csv("segment_labels/" + video_basename.split('.')[0] + ".csv")
