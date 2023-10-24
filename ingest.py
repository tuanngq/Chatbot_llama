from langchain.document_loaders import PyPDFLoader, DirectoryLoader   # Để tải, đọc văn bản (không cấu trúc)
from langchain.text_splitter import RecursiveCharacterTextSplitter   # Để tách văn bản
from langchain.embeddings import HuggingFaceEmbeddings   # Mã hóa văn bản
from langchain.vectorstores import FAISS   # Vector store


DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstores/db_faiss" # Tất cả embedding sẽ được lưu trong này

# Create vector database
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap = 50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2"
                                          , model_kwargs = {'device': 'cpu'})
    
    # embeddings = HuggingFaceEmbeddings(model_name = "keepitreal/vietnamese-sbert"
    #                                     , model_kwargs = {'device': 'cpu'})
    
    # embeddings = HuggingFaceEmbeddings(model_name = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
    #                                     , model_kwargs = {'device': 'cpu'})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()


