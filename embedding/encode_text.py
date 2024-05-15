from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from qdrant_client import QdrantClient

class EmcodedTranscriptpionVectorStore:
    def __init__(self, emd_model_name, collection_name):
        self.collection_name = collection_name
        self.qdrant_local_path = "/tmp/local_qdrant"
        self.qdrant_client = QdrantClient()
        self.emd_model = HuggingFaceEmbeddings(
            model_name=emd_model_name,
            model_kwargs={
                'trust_remote_code': True
            }
        )
        self.vector_store = Qdrant(
            client=self.qdrant_client, collection_name=self.collection_name, 
            embeddings=self.emd_model,
        )

    def embeddings_transcription(self, transcription_data_path: str, collection_name: str, db: QdrantClient):
        try:
            raw_documents = TextLoader(transcription_data_path).load()
            text_splitter = CharacterTextSplitter(
                separator="\n\n",
                is_separator_regex=False,
            )
            documents = text_splitter.split_documents(raw_documents)
            _ = self.vector_store.add_documents(documents)

        except Exception as ex:
            print(ex)

def test_query():
    pass
    
if __name__ == "__main__":
    test_transcription_path = '../data/state_of_union.csv'
    test_query = "covid-19"
    model_name = "Alibaba-NLP/gte-large-en-v1.5"
    collection_name = 'transcription_state_of_union'

    vstore = EmcodedTranscriptpionVectorStore(emd_model_name=model_name, collection_name=collection_name)
    docs = vstore.vector_store.similarity_search(test_query)
    print(docs[0].page_content)
