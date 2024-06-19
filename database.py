# ChromaDB
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
# from dotenv import load_dotenv
import os
import shutil
# load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']


CHROMA_PATH = "chroma"
DATA_PATH = "data/books"
BATCH_SIZE = 100

def main():
    generate_data_store()


def generate_data_store():
    docs = load_docs()
    chunks = split_text(docs)
    save_to_chroma(chunks)


def load_docs():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    return  loader.load()

def split_text(docs):
    text_split = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True
    )

    chunks = text_split.split_documents(docs)
    print(f"Split {len(docs)} documents into {len(chunks)} chunks")
    document = chunks[10]
    print(document.page_content)
    print(document.metadata)
    return chunks


def save_to_chroma(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


    # db = Chroma(persist_directory=CHROMA_PATH)

    # # Process in batches
    # for i in range(0, len(chunks), BATCH_SIZE):
    #     batch = chunks[i:i + BATCH_SIZE]
    #     db.add_documents(
    #         batch, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    #     )
    
    db = Chroma.from_documents(
        chunks[0:165], OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()

