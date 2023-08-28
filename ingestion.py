from dotenv import load_dotenv
load_dotenv()
import os

# Import the ReadTheDocsLoader for loading the documents
from langchain.document_loaders import ReadTheDocsLoader
# Import the RecursiveCharacterTextSplitter for splitting the documents
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import the OpenAIEmbeddings for generating embeddings
from langchain.embeddings import OpenAIEmbeddings

# Import the Pinecone for storing the vectors
from langchain.vectorstores import Pinecone

# Import the Pinecone SDK
import pinecone

pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENVIRONMENT_REGION"])

def ingest_docs() -> None:
    # Create a loader
    loader = ReadTheDocsLoader(path="langchain-docs/langchain-docs/langchain.readthedocs.io/en/latest")
    
    # Create a list of raw documents
    raw_documents = loader.load()

    # Print the number of documents
    print(f"loaded {len(raw_documents) } documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])

    # Split the documents
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(documents)} chunks")

    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to insert {len(documents)} to Pinecone")

    # Create an instance of the OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()

    # Create a Pinecone instance
    Pinecone.from_documents(documents, embeddings, index_name="langchain-doc-index")

    print("****** Added to Pinecone vectorstore vectors")

if __name__ == '__main__':
    ingest_docs()