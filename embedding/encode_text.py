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
        self.qdrant_local_path = "local_qdrant"
        self.emd_model = HuggingFaceEmbeddings(
            model_name=emd_model_name,
            model_kwargs={
                'trust_remote_code': True
            }
        )
        if not qdrant_client.collection_exists(self.collection_name):
            self.vector_store = Qdrant(
                client=QdrantClient(path=qdrant_local_path), 
                collection_name=self.collection_name, 
                embeddings=self.emd_model,
            )
        else:
            self.vector_store = Qdrant.from_existing_collection( 
                local=self.qdrant_local_path,
                collection_name=self.collection_name, 
                embeddings=self.emd_model,
            )

    def embeddings_transcription(self, transcription_data_path: str, collection_name: str):
        try:
            raw_documents = TextLoader(transcription_data_path).load()
            text_splitter = CharacterTextSplitter(
                separator="\n\n",
                is_separator_regex=False,
            )
            documents = text_splitter.split_documents(raw_documents)
            _ = self.vector_store.add_documents(documents)
            self.vector_store = Qdrant.from_documents(
                documents,
                self.emd_model,
                path=self.qdrant_local_path,
                collection_name=collection_name
            )

        except Exception as ex:
            print(ex)

def test_query():
    pass
    
if __name__ == "__main__":
    test_transcription_path = '../state_of_union.txt'
    test_query = "covid-19"
    model_name = "Alibaba-NLP/gte-large-en-v1.5"
    collection_name = 'transcription_state_of_union'

    vstore = EmcodedTranscriptpionVectorStore(emd_model_name=model_name, collection_name=collection_name)
    vstore.embeddings_transcription(test_transcription_path, collection_name)
    # docs = vstore.vector_store.similarity_search(test_query)
    # print(docs[0].page_content)
