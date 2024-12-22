import json
import os
from dotenv import load_dotenv
import streamlit as st
    # set streamlit page config
st.set_page_config(page_title="Chat with Documents",page_icon="📚",layout="centered")
# set the title
st.title("Chat with Documents")
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# load the env variables

# get the working directory
working_dir=os.path.dirname(os.path.abspath(__file__))
# Load the data from the json file
with open(os.path.join(working_dir, "config.json"), "r") as file:
    data = json.load(file)
GROQ_API_KEY = data["GROQ_API_KEY"]
# save the API key to the environment variable
os.environ["GROQ_API_KEY"]  = data["GROQ_API_KEY"]
# create the document loader
def load_documents(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        return loader.load()

def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter()
    doc_chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstore

def create_chain(vector_store):
    llm=ChatGroq(model='mixtral-8x7b-32768')

    retriever=vector_store.as_retriever()
    memory=ConversationBufferMemory(llm=llm,
    output_key="answer",
    memory_key="chat_history",
    return_messages=True)

    chain=ConversationalRetrievalChain.from_llm(llm=llm,
    retriever=retriever,
    memory=memory,
    chain_type="map_reduce")
    return chain

# initialize the streamlit chat_history
if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]
    # upload the document
uploaded_file=st.file_uploader(label="Upload a pdf  document",type=["pdf"])

if uploaded_file:
    file_path=f"{working_dir}/{uploaded_file.name}"	
    with open(file_path,"wb") as f:
        f.write(uploaded_file.getbuffer())

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = setup_vectorstore(load_documents(file_path))

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

    
    # chat with the document
user_input=st.chat_input("Ask a question")
    

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = st.session_state.conversation_chain({"question": user_input})
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
