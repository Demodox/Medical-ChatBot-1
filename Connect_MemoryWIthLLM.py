from langchain_huggingface import HuggingFaceEndpoint

from langchain_core.prompts import PromptTemplate 

from langchain.chains import RetrievalQA 

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import os
# STEP 1: Set up LLM (mistral with HuggingFace)
HF_TOKEN = os.environ.get("HF_TOKEN")
huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"


def load_llm(huggingface_repo_id):
    """
    Load the Hugging Face model.
    """
    llm = HuggingFaceEndpoint(
    repo_id=huggingface_repo_id,
    temperature=0.5,
    model_kwargs={"max_new_tokens": 512},
    huggingfacehub_api_token=HF_TOKEN
)

    return llm


#STEP 2: Connect LLM with FAISS and create chain


custom_prompt = """
You are a helpful assistant. Answer the question based on the provided context.
Use the context to provide a detailed and accurate answer. If the context does not contain enough information, say "I don't know".

context: {context}
question: {question}
start answer directly with "Answer: "

"""
def custom_prompt_template(prompt_temp):
    """
    Create a custom prompt template.
    """
    prompt=PromptTemplate(template=prompt_temp,input_variables=["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH = "vectorStore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(huggingface_repo_id),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt_template(custom_prompt)}
)

# query
user_query = input("Enter your question: ")
result = qa_chain.invoke({"query": user_query})

# Display the result
print("Answer:", result['result'])  
print("Source Documents:", result['source_documents'])

