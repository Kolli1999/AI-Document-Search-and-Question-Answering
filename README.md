**AI Document Search and Question Answering**

This repository contains a project that leverages LangChain, FAISS, and Groq to create a document search and question-answering system. The project allows users to upload documents, split them into manageable chunks, and perform semantic searches based on user queries. It also integrates with Groq’s language model for answering questions related to the content of the documents.

**Features**

Document Loading: Load text documents for processing.

Text Wrapping: Preserve the formatting of the loaded text for readability.

Text Splitting: Split documents into smaller chunks to optimize the search and answer retrieval process.

Vector Store with FAISS: Create a vector store using FAISS for efficient similarity search.

Question Answering with Groq: Use Groq’s language model to answer questions based on the similarity search results.

**Prerequisites**

Python 3.x

Google Colab or a local Python environment

An API key from Groq

**Installation**

Clone the repository:

git clone https://github.com/Kolli1999/AI-Document-Search-and-Question-Answering.git

Navigate to the project directory:

cd AI-Document-Search-and-Question-Answering

Install the required Python packages:

pip install -q langchain langchain_core langchain_community sentence_transformers faiss-cpu unstructured chromadb Cython tiktoken unstructured[local-inference] langchain_groq

**Usage**

**Load the Groq API Key:**

import getpass

import os

if "GROQ_API_KEY" not in os.environ:

    os.environ["GROQ_API_KEY"] = getpass.getpass("Provide your GROQ API TOKEN")
    
**Load and Process Documents:**

from langchain.document_loaders import TextLoader

loader = TextLoader('/path/to/your/document.txt')

documents = loader.load()

**Wrap Text for Readability:**

import textwrap

def wrap_text_preserve_newlines(text, width=110):

    lines = text.split('\n')
    
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    
    wrapped_text = '\n'.join(wrapped_lines)
    
    return wrapped_text

print(wrap_text_preserve_newlines(str(documents[0])))

**Split Documents into Chunks:**

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

docs = text_splitter.split_documents(documents)

**Create a Vector Store with FAISS:**

from langchain.embeddings import HuggingFaceEmbeddings

from langchain.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings()

db = FAISS.from_documents(docs, embeddings)

**Perform a Similarity Search:**

query = "explain about indian premier league"

docs = db.similarity_search(query)

print(wrap_text_preserve_newlines(str(docs[0].page_content)))

**Question Answering with Groq:**

import langchain_groq

from langchain_groq import ChatGroq

from langchain.chains.question_answering import load_qa_chain

GROQ_LLM = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="gemma2-9b-it")

chain = load_qa_chain(GROQ_LLM, chain_type="stuff")

query = "HOW IS INDIAN ECONOMY"

docs = db.similarity_search(query)

chain.run(input_documents=docs, question=query)

**Run Additional Queries:**

query = "explain about indian geography?"

docs = db.similarity_search(query)

chain.run(input_documents=docs, question=query)

**Customization**

Text Splitting: Adjust the chunk_size and chunk_overlap parameters in CharacterTextSplitter for different document types.

Groq Model: You can experiment with different Groq models by specifying the model parameter in ChatGroq.

Vector Store: You can replace FAISS with other vector store implementations if needed.

**License**

This project is licensed under the MIT License.

