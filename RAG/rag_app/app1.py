import certifi  # Provides Mozilla's CA bundle for SSL verification
import os  # OS utilities for environment variables and paths
import streamlit as st  # Streamlit for building the web UI
import time  # Time utilities (not used heavily here but commonly available)
from langchain_openai import OpenAI  # LLM wrapper for OpenAI-compatible models
# Loader for web pages
from langchain_community.document_loaders import UnstructuredURLLoader
# Text splitter utility
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # Chroma vectorstore wrapper
# Embeddings implementation using OpenAI
from langchain_openai import OpenAIEmbeddings
# High-level RAG chain creator
from langchain_classic.chains import create_retrieval_chain
# Chain to combine docs
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# Prompt template helper for chat-style prompts
from langchain_core.prompts import ChatPromptTemplate
# Generic web loader (alternate)
from langchain_community.document_loaders import WebBaseLoader

from dotenv import load_dotenv  # Loads env vars from a .env file
load_dotenv()  # Populate environment variables from .env if present


# Ensure the Python SSL layer uses certifi's CA bundle (macOS/conda compatibility)
os.environ["SSL_CERT_FILE"] = certifi.where()


# Set a custom user-agent string so some websites accept automated requests
os.environ["USER_AGENT"] = "MyLangChainBot/1.0 (contact: myemail@example.com)"


st.title("RAG App")  # Set the Streamlit app title

# Define the list of URLs to load and index
urls = ['https://www.friscotexas.gov/']
# Example: multiple sites can be provided
# urls = ['https://www.littleelm.gov/', 'https://www.friscotexas.gov/']

# Create a loader that fetches and parses the pages at the given URLs
loader = UnstructuredURLLoader(urls=urls)
# Load returns a list of Document-like objects (page content + metadata)
data = loader.load()

print(data)  # Debug: print loaded documents to console


# Create a RecursiveCharacterTextSplitter instance to break long text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
# Split the loaded documents into smaller chunks suitable for embedding
docs = text_splitter.split_documents(data)

# Keep the split document list (alias for clarity)
all_splits = docs
# Build a Chroma vectorstore from the split documents using OpenAI embeddings
vectorstore = Chroma.from_documents(
    documents=all_splits, embedding=OpenAIEmbeddings())

# Convert the vectorstore into a retriever for nearest-neighbor search
retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 6})

# Instantiate the LLM to use for generating answers
llm = OpenAI(model_name="gpt-4o-mini", temperature=0.4, max_tokens=500)

# Get user input from the Streamlit chat input widget (returns None if empty)
query = st.chat_input("Enter your query:")
# Keep a local variable `prompt` for later use (this will be replaced by template)
prompt = query

# System prompt template that instructs the assistant how to use retrieved context
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Build a chat-style prompt template with system + human message placeholders
prompt = ChatPromptTemplate.from_messages(
    [
        # System instructions with {context} placeholder
        ("system", system_prompt),
        # Human input placeholder that will be filled with query
        ("human", "{input}"),
    ]
)


if query:  # Only run retrieval/answering if the user provided a query
    # Create a chain that stuffs documents into the prompt and calls the LLM
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    # Create an end-to-end retrieval-augmented generation (RAG) chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Invoke the chain with the user's input and get the response dict
    response = rag_chain.invoke({"input": query})
    # Optionally: print(response["answer"])  # Debug to console

    # Display the generated answer in the Streamlit app
    st.write(response["answer"])
