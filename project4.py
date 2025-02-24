from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_PATH = 'data/'
DB_CHROMA_PATH ='vectorstore/db_chroma'

#creating vector database
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    
    documents=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=50)
    texts=text_splitter.split_documents(documents)

    embeddings= HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                      model_kwargs={'device': 'cpu'})
    
    db = Chroma(embedding_function=embeddings, persist_directory=DB_CHROMA_PATH)
    db.persist()

if __name__ == "__main__":
    create_vector_db()


