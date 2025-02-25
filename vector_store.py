from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import config  # Import API keys

# Load and process PDF into text chunks
def load_and_process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = text_splitter.split_documents(pages)
    return chunks

# Initialize vector store (ChromaDB)
def create_chroma_vector_store(chunks):
    embed = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=config.OPENAI_API_KEY)
    vector_store = Chroma.from_documents(chunks, embed)
    return vector_store

# Initialize Pinecone Vector Store
def create_pinecone_vector_store(chunks, index_name="custom-document-qa"):
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    
    # Create Pinecone index if not exists
    if index_name not in [idx["name"] for idx in pc.list_indexes()]:
        pc.create_index(index_name, dimension=1536, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))

    # Connect to the existing index
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=OpenAIEmbeddings())
    vectorstore.add_documents(documents=chunks)

    return vectorstore
