from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

async def loadDataAndCreateChunks(filename):
    file_path = os.path.join(filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    loader = PyPDFLoader(f"{file_path}") 
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
    text_chunks = text_splitter.split_documents(data)
    return text_chunks