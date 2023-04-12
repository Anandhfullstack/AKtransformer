from pydub import AudioSegment
import os


def segmentAudio(rootpath,audio_files, duration=4):

    for audio_file in audio_files:
        if audio_file.endswith('.mp3'):
            audio = AudioSegment.from_mp3(rootpath + audio_file)
            # Set duration for each segment (in milliseconds)
            segment_duration = duration * 1000  # 3 seconds
            # Calculate total number of segments
            total_segments = len(audio) // segment_duration

            for i in range(total_segments):
                # Calculate start and end time for current segment
                start_time = i * segment_duration
                end_time = (i + 1) * segment_duration
                # Extract segment from audio file
                segment = audio[start_time:end_time]
                # Save segment as WAV file
                audio_file=audio_file.replace('.mp3','')
                # print(audio_file+f'segment_{i + 1}.wav')
                # break
                export_path="D:/Projects/AKtransformer/bgm_audio/"+audio_file
                print(export_path+f'segment_{i + 1}.wav')
                segment.export(export_path+f'segment_{i + 1}.wav', format='wav')


# Loop through each segment

rootfolder ="D:/Projects/AKtransformer/"
folder_path = rootfolder +'data_analysis/'
file_names = os.listdir(folder_path)

segmentAudio(folder_path,file_names)