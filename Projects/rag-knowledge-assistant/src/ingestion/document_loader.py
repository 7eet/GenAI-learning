from langchain_community.document_loaders import (PyPDFLoader, TextLoader)

from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_document(file_path: str):
    extension = file_path.split(".")[-1].lower()
    
    if extension == 'pdf':
        loader = PyPDFLoader(file_path)
    elif extension in ['txt','md']:
        loader = TextLoader(file_path, 'utf-8')
    else:
        raise ValueError(f"Unsupported file type: {extension}")
    return loader.load();


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    
    return splitter.split_documents(documents)

