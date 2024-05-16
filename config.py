from langchain_community.embeddings import HuggingFaceEmbeddings

print("Downloading embedding")

EMB_MODEL = HuggingFaceEmbeddings(
                model_name="Alibaba-NLP/gte-large-en-v1.5",
                model_kwargs={
                    'trust_remote_code': True
                }
            )