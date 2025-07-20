from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS




#step 1: Load the PDF file
DATA_PATH = "data/"
def load_pdf(data):
    loader = DirectoryLoader(
        data, glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

document = load_pdf(DATA_PATH)
print("length of documents:", len(document))

#step 2: Split the documents into smaller chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs

split_docs = split_documents(document)
print("length of split documents:", len(split_docs))

#step 3: Create embeddings for the split documents
def create_embeddings():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model

embedding_model = create_embeddings()


# store embeddings in Faiss
DB_FAISS_PATH = "vectorStore/db_faiss"
db=FAISS.from_documents(split_docs,embedding_model)
db.save_local(DB_FAISS_PATH)