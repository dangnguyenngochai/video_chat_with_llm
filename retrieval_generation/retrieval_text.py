import langchain_community 
import os
import getpass
from embedding import (
    EmcodedTranscriptpionVectorStore, 
    test_run as dummy_emb,
    )

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub

# configuring google api
def connect_gemini(key):
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass(key)

def prompt_generator(context, query):
    pass
    

def generate_response(vstore: EmcodedTranscriptpionVectorStore, query: str): 
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # fetch prompt template
    prompt = hub.pull("rlm/rag-prompt")
    retriever = vstore.get_retriever(5)
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    response = rag_chain.invoke(query)
    return response

GOOGLE_STUDIO_KEY = 'AIzaSyB-PQLnrQm2Z5UGXW28R24TXG99MLPICFw'
connect_gemini(GOOGLE_STUDIO_KEY)

def test_run():
    query = "What did America do during Covid-19 to migitate its impact ?"
    dummy_vt = dummy_emb()
    response = generate_response(dummy_vt, query)
    print(response)

if __name__ == '__main__':
    test_run()