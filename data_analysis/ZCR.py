import os

import numpy as np
import pandas as pd
import librosa
import time
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd



def plot_ZCR(s1_ZCR, s2_ZCR, mix_clean_ZCR, times):
    fig, ax = plt.subplots()
    ax.plot(times, s1_ZCR.T, label="Speaker 1", color='black')
    ax.plot(times, s2_ZCR.T, label="Speaker 2", color='green')
    ax.plot(times, mix_clean_ZCR.T, label="Mixed", color='red')
    ax.legend(loc='upper right')
    ax.set(title='log Power spectrogram')


rootfolder ="D:/Projects/DANet-For-Speech-Separation-master/data/Libri2mixSimply/"
folder_path = rootfolder +'/train/mix_clean'
file_names = os.listdir(folder_path)

speaker_1=[]
speaker_2=[]
mixed = []
times=0
start_time = time.time()



for filename in file_names:
    elapsed_time = time.time() - start_time


    s1, s1sr = librosa.load(rootfolder +'/train/s1/'+filename, sr=8000, duration= 3.0)
    s2, s2sr = librosa.load(rootfolder + '/train/s2/' + filename, sr=8000, duration= 3.0)
    mix_clean, mix_cleansr = librosa.load(rootfolder + '/train/mix_clean/' + filename, sr=8000, duration= 3.0)
    s1_ZCR = librosa.feature.zero_crossing_rate(s1)
    s2_ZCR = librosa.feature.zero_crossing_rate(s2)
    mix_clean_ZCR = librosa.feature.zero_crossing_rate(mix_clean)
    speaker_1.append(s1_ZCR)
    speaker_2.append(s2_ZCR)
    mixed.append(mix_clean_ZCR)
    times = librosa.times_like(s1_ZCR)
    plot_ZCR(s1_ZCR, s2_ZCR, mix_clean_ZCR, times)

    # librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    #                          y_axis='log', x_axis='time', ax=ax)
    #
    # break



    # ax.plot(times, trump_cent.T, label='trumpet centroid', color='blue')


    if elapsed_time >= 5:
        break

    # print(ZCR.shape)
    # break;



mixed = np.array(mixed)
speaker_1 = np.array(speaker_1)
speaker_2 = np.array(speaker_2)
mn=mixed.shape[0]*mixed.shape[2]
rs_mixed = mixed.reshape(mn)
rs_speaker_1 = speaker_1.reshape(mn)
rs_speaker_2 = speaker_2.reshape(mn)


# plt.legend(visible=False)

# plt.show()


import numpy as np
# import matplotlib.pyplot as dlt
#
# # Generate random height and weight data for 100 individuals
# # height = np.random.normal(170, 10, 100)
# # weight = height * np.random.normal(0.01, 0.02, 100) + np.random.normal(70, 10, 100)
#
# # Create a scatter plot of height vs weight
# plt.scatter(rs_speaker_1, rs_speaker_2)
#
# # Set the plot title and axis labels
# dlt.title('Height vs Weight Scatter Plot')
# dlt.xlabel('Height (cm)')
# dlt.ylabel('Weight (kg)')
#
# # Show the plot
# dlt.show()

#this will plot the spectral centroid of both
# sy, sr = librosa.load("s2.wav",duration=5.0)
# sz, sri = librosa.load("s1.wav",duration=5.0)
# bothy, bothsr = librosa.load("both.wav",duration=5.0)
#
# trump, tsr = librosa.load(librosa.ex('trumpet'),duration=5.0)
# y_8k = librosa.resample(y, orig_sr=sr, target_sr=8000)
# mfcc = librosa.feature.mfcc(y=sy, sr=sr)
# cent = librosa.feature.spectral_centroid(y=sy, sr=sr)
# centi = librosa.feature.spectral_centroid(y=sz, sr=sr)
# bothy_cent= librosa.feature.spectral_centroid(y=bothy, sr=sr)
# trump_cent= librosa.feature.spectral_centroid(y=trump, sr=tsr)
# S, phase = librosa.magphase(librosa.stft(y=sy))
# freqs, times, D = librosa.reassigned_spectrogram(y, fill_nan=True)
# librosa.feature.spectral_centroid(S=np.abs(D), freq=freqs)
# times = librosa.times_like(cent)




#
#
# data = {'Mixed ZCR': rs_mixed,
#         'Speaker 2 ZCR': rs_speaker_1,
#         'Speaker 3 ZCR': rs_speaker_2}
#
# # Create a DataFrame from the data
# df = pd.DataFrame(data)
#
# # Create the violin plot
# sns.violinplot(data=df, inner='stick')
#
# # Set the title and axis labels
# plt.title('ZCR Comparison between Speakers 1 , 2 and mixed audio')
# plt.xlabel('Speaker')
# plt.ylabel('ZCR')
#
# # Show the plot
# plt.show()

# df = pd.DataFrame(mixed, columns=['mixed'])
# df.to_excel('output.xlsx', index=False)