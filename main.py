print('testing out whisper openai api')

# from openai import OpenAI

from preprocess import (
    extract_transcription
    # extract_frames
    )
import os 

#preparing frames
def extract_frames(timespan: list = None):
    if timespan is not None:
        pass
    print("working")
    
if __name__ == "__main__":
    test_file_path = 'data/video/cX4DUogRjso.mp4'
    transcriptions = extract_transcription(test_file_path)
    _ = extract_frames()