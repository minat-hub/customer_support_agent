import streamlit as st
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

st.title("Customer Support AI Assistant")
cleaned_file = "sample_customer_data.csv"
if cleaned_file is not None:

    df = pd.read_csv(cleaned_file)

    if "question" in df.columns and "answer" in df.columns:

        texts = []
        for _, row in df.iterrows():
            texts.append(f"QUESTION: {row['question']}\nANSWER: {row['answer']}")

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.create_documents(texts)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

        vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
        # You can switch between OllamaLLM and ChatGroq here
        llm = ChatGroq(model="openai/gpt-oss-120b")
        # llm = OllamaLLM(model="llama3.2")

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=vector_store.as_retriever()
        )

        st.subheader("What issue are you facing?")
        question = st.text_input("Enter your question here:")

        if question:
            with st.spinner("Searching for solutions..."):
                answer = qa_chain.invoke(question)
                st.success("**Answer:**")
                st.write(answer["result"])
                # If ollama is used, uncomment the line below
                # st.write(answer)
    else:
        st.error("CSV must contain 'question' and 'answer' columns")
else:
    st.info("Please upload a CSV file with your email data to begin.")


st.divider()
st.caption(
    "Source: [github.com/minat-hub/customer_support_agent](https://github.com/minat-hub/customer_support_agent)"
)
