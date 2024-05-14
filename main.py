print('testing out whisper openai api')

from openai import OpenAI

import moviepy.editor as movie_editor

import os 

os.environ['OPENAI_API_KEY'] = ''

def extract_audio(audio_file_path):
    
    # OpenAI read the keys in os.envrion
    client = OpenAI()
    # client.api_key = os.environ['OPENAI_API_KEY']

    audio_file= open(audio_file_path, "rb")
    transcription = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file
    )

    print(transcription.text)

if __name__ == "__main__":
    test_file_path = 'C:\\Users\\DANG\\Downloads\\cX4DUogRjso.mp4'

    if os.path.exists(test_file_path):
        print('Asmongold is yapping')
        video = movie_editor.VideoFileClip(test_file_path)
        audio = video.audio
        audio_file = test_file_path.split('\\')[-1].split('.')[0]
        audio_file_path = f'data\\audio\\{audio_file}.mp3'.format(audio_file)
        audio.write_audiofile(audio_file_path, codec="libmp3lame", bitrate="320k")
    else:
        print("Asmongold is balding")

    transcriptions = extract_audio(audio_file_path)
    print(transcriptions)