from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from qdrant_client import QdrantClient

test_transcription_path = '../data/transcription/cX4DUogRjso.csv'
emd_model = "Alibaba-NLP/gte-large-en-v1.5"


def embeddings_data(transcription_data_path):

    raw_documents = TextLoader(transcription_data_path).load()
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        is_separator_regex=False,
    )
    documents = text_splitter.split_documents(raw_documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=emd_model,
        model_kwargs={
            'trust_remote_code': True
        }
    )

    db = Qdrant.from_documents(documents, embeddings, location=":memory:", collection_name="my_documents")
    return db
    
def test_query():
    pass
    
if __name__ == "__main__":
    query = "pirate on sony"
    docs = db.similarity_search(query)
    embeddings_data = 
    print(docs[0].page_content)
