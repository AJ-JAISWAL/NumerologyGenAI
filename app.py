from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from pymongo.mongo_client import MongoClient
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
import streamlit as st

qrok_api_key="xxxxxxxxxxxx"

uri = "xxxxx"
client = MongoClient(uri)
db = client["xxxx"]
collection = db["yyyy"]


def load_llm():
    llm = ChatGroq(groq_api_key=qrok_api_key,
                   model_name="Llama3-8b-8192")
    return llm


def retrieval_qa_chain(llm, prompt,vector_store):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain


def qa_bot():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embedding_model,
        index_name="vector_index_numerology",
        relevance_score_fn="cosine",
    )
    llm = load_llm()
    qa = retrieval_qa_chain(llm,query, vector_store)
    return qa


def answer_query(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

                                                         #App Code
st.title('🤖🧠 Tantralogy')
st.write('This chatbot will provide the knowledge about Numerology.')

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to the Tantralogy ask anything related to Numerology?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to the Tantralogy ask anything related to Numerology?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response= answer_query(prompt)['result']
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
