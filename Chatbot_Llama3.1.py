import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS 
from langchain_community.document_loaders import PyPDFDirectoryLoader

groq_api_key = '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'

st.title("Welcome to Textbook Chatbot Powered by Llama3.1")

llm=ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device':'cpu'})

def vector_embedding():

    if "vectors" not in st.session_state:

        st.session_state.embeddings = embeddings
        st.session_state.loader=PyPDFDirectoryLoader("./data")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        st.session_state.vectors= FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)





prompt1= st.text_input("Enter your Question from Document")


if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB is Ready")

import time



if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time:",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("--------------------------------")