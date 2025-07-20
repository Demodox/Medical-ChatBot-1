import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS



DB_FAISS_PATH = "vectorStore/db_faiss"
@st.cache_resource
def get_vector_store():
    """
    Load the FAISS vector store with HuggingFace embeddings.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    

def set_custom_prompt(template_text):
    return PromptTemplate(template=template_text, input_variables=["context", "question"])

# Setup Google Gemini LLM
def load_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  
        temperature=0.5
    )
    return llm


def main():
    st.title("MediBot - Your Medical Assistant")
    st.subheader("Welcome to MediBot, your personal medical assistant powered by AI.")

    # Initialize session state for messages
    if 'message' not in st.session_state:
        st.session_state.message = []
    # Display chat messages
    for message in st.session_state.message:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Enter your medical question or query:")
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.message.append({'role':'user', 'content': prompt})

        # Set custom prompt template
        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        Don't provide anything outside the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """
        custom_prompt = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)

        # Load vector store and create QA chain
        try:
            db = get_vector_store()
        except Exception as e:
            st.error(f"Error loading vector store: {e}")
            return

        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(),
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': custom_prompt}
        )

        # Get the answer from the chain
        response = qa_chain.invoke({'query': prompt})
        source_docs = response.get("source_documents", [])
        

        # Display the result
        st.chat_message('assistant').markdown(response["result"])
        st.session_state.message.append({'role':'assistant', 'content': response["result"]})
        if source_docs:
            st.markdown("### 📄 Source Documents:")
            for doc in source_docs:
                st.markdown(f"- {doc.metadata.get('source', 'Unknown Source')}: {doc.page_content[:200]}...")
if __name__ == "__main__":
    main()