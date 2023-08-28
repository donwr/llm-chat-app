from dotenv import load_dotenv
load_dotenv()
import os 
from langchain.embeddings import OpenAIEmbeddings

# Import Chat Model for the chatbot
from langchain.chat_models import ChatOpenAI

# Import RetrievalQA for the retrieval question answering
from langchain.chains import RetrievalQA

from langchain.vectorstores import Pinecone
import pinecone

INDEX_NAME="langchain-doc-index"

pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENVIRONMENT_REGION"])


def run_llm(query:str):
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)
    return qa({"query": query})

if __name__ == '__main__':
    print(run_llm("What is Langchain?"))