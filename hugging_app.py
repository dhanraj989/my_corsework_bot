import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

st.title("Coursework Q&A")

groq_api_key = st.secrets["GROQ_API_KEY"]

if not groq_api_key:
    st.error("Please set your GROQ_API_KEY as an environment variable or in Streamlit secrets.")
    st.stop()

@st.cache_data(show_spinner=True)
def load_data_and_build_vectorstore():
    # Load the PDFs from the folder
    loader = PyPDFDirectoryLoader("./data_source")
    documents = loader.load()

    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(documents)

    # Embedding Using HuggingFace BGE Embeddings
    huggingface_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-l6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # VectorStore Creation
    vectorstore = FAISS.from_documents(final_documents, huggingface_embeddings)
    return vectorstore

@st.cache_resource(show_spinner=True)
def load_llm():
    # Use ChatGroq as LLM (non-serializable resource)
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="Llama3-8b-8192"
    )
    return llm

with st.spinner("Loading data and building vectorstore..."):
    vectorstore = load_data_and_build_vectorstore()

with st.spinner("Loading language model..."):
    llm = load_llm()

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

prompt_template = """
Use the following piece of context to answer the question asked.
Please try to provide the answer only based on the context

{context}
Question:{question}

Helpful Answers:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])

retrievalQA = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

st.write("Ask question:")
query = st.text_input("Enter your query", "")

if st.button("Get Answer") and query.strip():
    with st.spinner("Processing your query..."):
        result = retrievalQA.invoke({"query": query})
        st.write("**Answer:**")
        st.write(result['result'])

        with st.expander("Source Documents"):
            for doc in result['source_documents']:
                st.write(doc.page_content)
                st.write("---")
