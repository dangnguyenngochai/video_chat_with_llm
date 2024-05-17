from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from transformers import CLIPProcessor, CLIPModel
import torch
from qdrant_client.http import models as rest
import os
# import torch.nn.functional as F
# from transformers import AutoModel, AutoTokenizer

from qdrant_client import QdrantClient
import pathlib
from .transcription_loader import (
    TranscriptionLoader
)

from PIL import Image
from langchain_community.embeddings import HuggingFaceEmbeddings

class EmcodedImgVectorStore:
    def __init__(self, 
                 collection_name, 
                 qdrant_client, 
                 model=None, 
                 emb_model_name="openai/clip-vit-base-patch32"
                 ):
        
        self.collection_name = collection_name
        self.qdrant_local_path = "local_qdrant"
        self.qdrant_client = qdrant_client

        if model is not None:
            self.emd_model = model
        else:
            self.emd_model = CLIPModel.from_pretrained(emb_model_name)
            self.processor = CLIPProcessor.from_pretrained(emb_model_name)
            
        if not qdrant_client.collection_exists(self.collection_name):
            pass        
        else:
            qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(size=512, 
                                                 distance=rest.Distance.COSINE
                                                 ),
            )

    def __extract_frames(video_path, output_folder):
        try:
            video_path = video_path
            frames_path = output_folder
            os.system(f'ffmpeg -i {video_path} -vf "fps=1" {frames_path}'.format(video_path, frames_path))
            print("Finished extracting frames")
        except Exception as ex:
            print("Good work")
            print(ex)

    def embedding_imgs(self, imgs_data_path: str, collection_name: str):
        try:            
            if self.qdrant_client.collection_exists(self.collection_name):
                print("Frames are indexed")
                return None

            for index, img_pth in enumerate(os.listdir(imgs_data_path)):
                with Image.open(img_pth) as img:
                    # Preprocess the image and generate embeddings
                    inputs = self.processor(images=img, return_tensors="pt")
                    embeddings = self.emd_model.get_image_features(**inputs).detach().numpy()
                    # Prepare the embeddings data
                    embedding_data = embeddings[0].tolist()  # Assuming batch size of 1 for simplicity
                    # Upload the embeddings to Qdrant
                    self.qdrant_client.upsert(
                        collection_name=collection_name,
                        points=[
                            rest.PointStruct(id=index, vector=embedding_data, payload={"image_path": img_pth})
                        ]
                    )
            
        except Exception as ex:
            print(ex)

    def query_relevants(self, img_query, top_k):
        # Example query
        inputs = self.processor(images=img_query, return_tensors="pt")
        embeddings = self.emd_model.get_image_features(**inputs).detach().numpy()
        results = self.qdrant_client.search(
            collection_name="image_collection",
            query_vector=embeddings,
            limit=1
        )
        img_pth = results.payload.get('image_path')
        return img_pth
    
def test_run() -> EmcodedImgVectorStore:
    test_video_path = 'data/video/cX4DUogRjso.mp4'
    test_query = 'test_images/yapping.png'
    # model_name = "Alibaba-NLP/gte-large-en-v1.5"

    import sys
    sys.path.append('../')
    from config import EMB_MODEL

    collection_name = 'asmon-yapping'

    try:
#         qdrant_client = QdrantClient(path='local_qdrant')
        qdrant_client = QdrantClient(location=':memory:')
        vstore = EmcodedImgVectorStore(collection_name=collection_name, qdrant_client=qdrant_client)
        
        # test embeddings
        vstore.embedding_imgs(test_video_path, collection_name)
        
        # test query
        print('Running test for querying the indexed data')
        img_path = vstore.query_relevants(test_query, 1)
        print(img_path)
        return vstore
    
    except Exception as ex:
        print(ex)

if __name__ == "__main__":
    test_run()