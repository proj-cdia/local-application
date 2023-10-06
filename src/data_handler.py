import os
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter

class DataHandler:
    def __init__(self, data_path):
        self.data_path = data_path
        self.loader = None
        self.documents = []
        self.documents_chunks = None
        self.vector_db = None
        self.sppliter_model = CharacterTextSplitter(separator='\n', chunk_size=500, chunk_overlap=100)
        self.embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    def load_data(self):
        for file in os.listdir(self.data_path):
            file_path = os.path.join(self.data_path, file)
            if file.endswith('.pdf'):
                self.loader = PyPDFLoader(file_path)
                self.documents.extend(self.loader.load())
            elif file.endswith('.txt'):
                self.loader = TextLoader(file_path)
                self.documents.extend(self.loader.load())
            elif file.endswith('.docx'):
                self.loader = Docx2txtLoader(file_path)
                self.documents.extend(self.loader.load())
        return self.documents
    
    def chunk_data(self):
        self.documents_chunks = self.sppliter_model.split_documents(self.documents)
        return self.documents_chunks
    
    def create_vector_db(self):
        self.vector_db = Chroma.from_documents(self.documents_chunks, self.embedding_model, persist_directory='./vector-data')
        self.vector_db.persist()
        return self.vector_db