print('testing out whisper openai api')

# from openai import OpenAI

from preprocess import (
    extract_transcription,
    extract_frames
    )
import os

from embedding import *
from retrieval_generation import *

from config import EMB_MODEL

def get_audio_transcription_path(video_file_path):
    name = video_file_path.split('/')[-1].split('.')[0]
    audio_file_path='data/audio/%s.mp3'%name
    transcription_file_path='data/transcription/%s.csv'%name
    return name, audio_file_path, transcription_file_path

def process_videos(video_file_path, query):
    try:        
        filename, audio_file_path, transcription_file_path = get_audio_transcription_path(video_file_path)
        print('Fetching files...Done')
        
        if not os.path.exists(transcription_file_path):
            extract_transcription(video_file_path)

        qdrant_client = QdrantClient(location=':memory:')
        collection_name = 'transcription_' + filename

        print("Indexing transcription")
        vstore = EmcodedTranscriptpionVectorStore(model=EMB_MODEL, collection_name=collection_name, qdrant_client=qdrant_client) 
        vstore.embeddings_transcription(transcription_file_path, collection_name)
        print("Indexing done")

        response = generate_response(vstore, query)
        
        print("Question:", query, '\n', "Answer:", response)
        return response
    
    except Exception as ex:
        print(ex)
        print("Nice try !!!")

if __name__ == "__main__":
    test_file_path = 'data/video/cX4DUogRjso.mp4'
    _= extract_transcription(test_file_path)
    _ = extract_frames()