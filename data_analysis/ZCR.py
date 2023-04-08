import os

import numpy as np
import pandas as pd
import librosa
import time
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd





rootfolder ="D:/Projects/DANet-For-Speech-Separation-master/data/Libri2mixSimply/"
folder_path = rootfolder +'/train/mix_clean'
file_names = os.listdir(folder_path)

speaker_1=[]
speaker_2=[]
mixed = []

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

    if elapsed_time >= 30:
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


data = {'Mixed ZCR': rs_mixed,
        'Speaker 2 ZCR': rs_speaker_1,
        'Speaker 3 ZCR': rs_speaker_2}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Create the violin plot
sns.violinplot(data=df, inner='stick')

# Set the title and axis labels
plt.title('ZCR Comparison between Speakers 1 , 2 and mixed audio')
plt.xlabel('Speaker')
plt.ylabel('ZCR')

# Show the plot
plt.show()

# df = pd.DataFrame(mixed, columns=['mixed'])
# df.to_excel('output.xlsx', index=False)