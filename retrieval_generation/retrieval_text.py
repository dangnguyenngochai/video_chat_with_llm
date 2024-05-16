import os
from embedding import (
    EmcodedTranscriptpionVectorStore, 
    test_run as dummy_emb,
    )

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# from langchain_google_genai import (
#     GoogleGenerativeAI
# )
from langchain import hub
from langchain_cohere import ChatCohere


# os.environ["OPENAI_API_KEY"] = 'sk-proj-0VuYdXfFXk6evfzOzLJTT3BlbkFJcshpzLyIT1ij7tR8q11Q'
# os.environ["GOOGLE_API_KEY"] = 'AIzaSyB-PQLnrQm2Z5UGXW28R24TXG99MLPICFw'
os.environ['COHERE_API_KEY'] = 'OhqrR0Bude8zr30XWVjreKC4LNbNHPhivxw7n0Vw'

def prompt_generator(context, query):
    pass
    
def generate_response(vstore: EmcodedTranscriptpionVectorStore, query: str): 
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # fetch prompt template
    prompt = hub.pull("rlm/rag-prompt")
    retriever = vstore.get_retriever(5)
    llm = ChatCohere(model="command-r")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    response = rag_chain.invoke(query)
    return response

def test_run2():
    query = "What did America do during Covid-19 to migitate its impact ?"
    dummy_vt = dummy_emb(run_query=False)
    response = generate_response(dummy_vt, query)
    print(response)

if __name__ == '__main__':
    test_run2()