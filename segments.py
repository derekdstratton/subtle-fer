from pytube import YouTube
import cv2
import numpy

# getting ffmpeg to work on wsl, building ope
# ncv for openface: https://github.com/justadudewhohacks/opencv4nodejs/issues/274
import os
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import numpy as np

def extract_features(video_path):
    csv_from_video_path = "processed/" + video_path.split('.')[0] + ".csv"
    if not os.path.isfile(csv_from_video_path):
        openface_build_path = "/mnt/c/Users/derek/PycharmProjects/OpenFace/build/bin"

        # for a single person
        os.system(openface_build_path + "/FeatureExtraction")

        # https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments
        subprocess.run([openface_build_path + "/FeatureExtraction", "-f", video_path])
    else:
        print("csv file already exists, loading into df")
    df = pd.read_csv(csv_from_video_path)
    return df

# one idea is to take a big rolling average (N=len(x)) as the baseline
# then compute the standard deviation. this makes assumptions that there's
# a distribution of the data that's centered around the mean, which may
# not be accurate... my goal is to detect anomalous signals
def plot_au_over_time(df, au_num, rolling_mean_window=20):
    plt.plot(df['frame'], df['AU' + f'{au_num:02}' + '_r'])
    x = df['AU' + f'{au_num:02}' + '_r']
    df['interesting_segments' + 'AU' + f'{au_num:02}'] = plot_with_rolling(df['frame'], x, rolling_mean_window)


    # # smooths it out a little bit.
    # rolling_mean = np.convolve(x, np.ones(rolling_mean_window) / rolling_mean_window, mode='same')
    #
    # # the full window gives something a bit clearer.
    # rolling_mean_with_full_window = np.convolve(x, np.ones(len(x)) / len(x), mode='same')
    #
    # # plot the moving average
    # plt.plot(df['frame'], rolling_mean)
    # plt.plot(df['frame'], rolling_mean_with_full_window)
    # plt.plot(df['frame'], rolling_mean_with_full_window + deviation)
    # plt.show()
    # # AU 6 deviates from the mean really nicely except for surprise
    # # we can maybe generalize it to go thru all the AUs, and find
    # # all segments that deviate from each's distribution of "normal"
    # # to find the emotions. 25 captures surpise well.
    # df['interesting_segments' + 'AU' + f'{au_num:02}'] = rolling_mean > rolling_mean_with_full_window + deviation
    # https://dsp.stackexchange.com/questions/48508/how-detect-signal-from-noise
    # https://en.wikipedia.org/wiki/Forecasting
    # https://en.wikipedia.org/wiki/Time_series
    # this can dive in deep
    # https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    # try plotting the overall mean and sd's, rolling mean
    # you could compare the actual values or rolling vals to determine sd
    # todo: try pca to get au combinations, rather than saying "for all AUs find the intersection of useful segments. which might be good idk"
    # it might be useful to characterize segments by the number of AU matches, and
    # the length of the segments (1 match may not be useful) (small segments may
    # or may not be useful). You may want to just manually delete segments that
    # show up though)

def plot_with_rolling(frame, au_over_time, rolling_mean_window):
    deviation = np.std(au_over_time)
    # smooths it out a little bit.
    rolling_mean = np.convolve(au_over_time, np.ones(rolling_mean_window) / rolling_mean_window, mode='same')

    # the full window gives something a bit clearer.
    # rolling_mean_with_full_window = np.convolve(au_over_time, np.ones(len(au_over_time)) / (len(au_over_time)), mode='same')
    rolling_mean_with_full_window = np.convolve(au_over_time, np.ones(len(au_over_time)//2) / (len(au_over_time)//2), mode='same')

    # https://stackoverflow.com/questions/46964363/filtering-out-outliers-in-pandas-dataframe-with-rolling-median


    # plot the moving average
    plt.plot(frame, rolling_mean)
    plt.plot(frame, rolling_mean_with_full_window)
    plt.plot(frame, rolling_mean_with_full_window + deviation)
    # plt.plot(df['frame'], rolling_mean)
    # plt.plot(df['frame'], rolling_mean_with_full_window)
    # plt.plot(df['frame'], rolling_mean_with_full_window + deviation)
    plt.show()
    deviation = np.std(au_over_time)
    return rolling_mean > rolling_mean_with_full_window + 2* deviation


def logic_on_au_features(df):
    # https://stackoverflow.com/questions/11285613/selecting-multiple-columns-in-a-pandas-dataframe
    # NOTE: i'm deciding to remove au45 since it's acting weird
    # (guess since i'm slicing this may also omit 26?)
    b=df.loc[:, 'interesting_segmentsAU01':'interesting_segmentsAU26']
    c=b.any(axis=1)
    df['interesting_segments'] = c
    plt.plot(df['frame'], c)
    plt.show()

# https://medium.com/@ansjin/dimensionality-reduction-using-pca-on-multivariate-timeseries-data-b5cc07238dc4
def pca_on_au_features(df):
    b=df.loc[:, 'AU01_r':'AU26_r']
    from sklearn.decomposition import PCA
    pca = PCA(n_components=4)
    return pca.fit_transform(np.array(b))