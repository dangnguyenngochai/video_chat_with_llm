from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

import torch.nn.functional as F
# from transformers import AutoModel, AutoTokenizer

from qdrant_client import QdrantClient
import pathlib
from transcription_loader import TranscriptionLoader

class EmcodedTranscriptpionVectorStore:
    def __init__(self, emd_model_name, collection_name,  qdrant_client):
        self.collection_name = collection_name
        self.qdrant_local_path = "local_qdrant"
        self.emd_model = HuggingFaceEmbeddings(
            model_name=emd_model_name,
            model_kwargs={
                'trust_remote_code': True
            }
        )
        
        if not qdrant_client.collection_exists(self.collection_name):
            self.vector_store = None
        else:
            self.vector_store = Qdrant.from_existing_collection( 
#                 path=self.qdrant_local_path,
                location = ':memory:',
                collection_name=self.collection_name, 
                embeddings=self.emd_model,
            )
            
    def __load_loader(self,ext):
        support_ext = {
            '.txt': TextLoader,
            '.csv': TranscriptionLoader
        }
        if support_ext.get(ext, False):
            return support_ext.get(ext)
        else:
            print('Transcription file format does not exist')
    
    def __load_transcription_segments(self,transcription_data_path: str) -> list[Document]:
        document_loader = self.__load_loader(pathlib.Path(transcription_data_path).suffix)
        
        raw_documents = document_loader(transcription_data_path).load()
        
        if document_loader is TextLoader:
            text_splitter = CharacterTextSplitter(
                separator="\n\n",
                is_separator_regex=False,
            )
            
            documents = text_splitter.split_documents(raw_documents)
        else:
            documents = raw_documents
        return documents
    
    def get_retriever(self, top_k):
        if self.vector_store is not None:
            # using cosine similarity
            retriever = self.vector_store.as_retriever(search_type="similarity",
                                                 search_kwargs={"k": top_k})
            return retriever
        else:
            print("The transcript has not been loaded into the vector database")

    def embeddings_transcription(self, transcription_data_path: str, collection_name: str):
        try:
            documents = self.__load_transcription_segments(transcription_data_path)
            if self.vector_store is not None:
                _ = self.vector_store.add_documents(documents)
            else:       
                self.vector_store = Qdrant.from_documents(
                    documents,
                    self.emd_model,
#                     path=self.qdrant_local_path,
                    location=':memory:',
                    collection_name=collection_name
                )
        except Exception as ex:
            print(ex)

    def query_relevants(self, query, top_k):
        # using cosine similarity
        retriever = self.get_retriever(top_k)
        if retriever is not None:
            relevants = retriever.invoke(query)
            return relevants

def test_run() -> EmcodedTranscriptpionVectorStore:
    test_transcription_path = 'video_chat_with_llm/state_of_union.txt'
    test_query = "What did America do during Covid-19 to migitate its impact ?"
    model_name = "Alibaba-NLP/gte-large-en-v1.5"
    collection_name = 'transcription_state_of_union'

    try:
#         qdrant_client = QdrantClient(path='local_qdrant')
        qdrant_client = QdrantClient(location=':memory:')
        vstore = EmcodedTranscriptpionVectorStore(emd_model_name=model_name, collection_name=collection_name, qdrant_client=qdrant_client)
        
        # test embeddings
        vstore.embeddings_transcription(test_transcription_path, collection_name)
        
        # test query
        print('Running test for querying the indexed data')
        relevants = vstore.query_relevants(test_query, 1)
        print(relevants)

        return vstore
    except Exception as ex:
        print(ex)

if __name__ == "__main__":
    test_run()