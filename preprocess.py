import whisper
import moviepy.editor as movie_editor

import os

# preparing transcription
def extract_transcription(video_file_path):
    def extract_audio(video_file_path):
        video = movie_editor.VideoFileClip(video_file_path)
        audio = video.audio
        audio_file = video_file_path.split('/')[-1].split('.')[0]
        audio_file_path = f'data/audio/{audio_file}.mp3'.format(audio_file)
        audio.write_audiofile(audio_file_path, codec="libmp3lame", bitrate="320k")
        return audio_file_path

    if os.path.exists(video_file_path):
        audio_file_path = extract_audio(video_file_path)
        model = whisper.load_model('large')
        transcription = model.transcribe(audio_file_path)
        transcription_file_path = 'data/transcription/%s.csv'%(audio_file_path.split('/')[-1].split('.')[0])
        
        #checkk transcription output
        print(transcription['text'])
        
        with open(transcription_file_path, 'w+') as file:
            data = []
            for seg in transcription['segments']:
                start = seg['start']
                end = seg['end']
                text = seg['text']
                rec = ','.join([start, end, text])
                data.append(rec)
            file.writelines(data)
    else:
        print("Video path not exist")

