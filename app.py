import fitz  # PyMuPDF
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# 1. load your document
pdf_path = "./movie_reviews.pdf" # your document path
doc = fitz.open(pdf_path)

documents = []

for page_num in range(len(doc)):
    page = doc[page_num]
    text = page.get_text()
    if text.strip():  # skip white space
        document = Document(
            page_content=text,
            metadata={"page": page_num + 1}  # mark your metadata in case you wanna know which pages have been used
        )
        documents.append(document)

doc.close()

#2. create embedding instance
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

#3. create chroma db to store your document
db_location = "./chroma_db" # location where you wanna store your db
vector_store = FAISS.from_documents(documents, embeddings)

#4. store document in vector store and keep renewing your db whenever you run this app
# vector_store.add_documents(documents)

#5. create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # set k base on how many docs to return

#6. setup streamlit interface
st.set_page_config(page_title="Streaming Agent", page_icon="🤖")
st.title("VietHa-knowyourdocs ♌")

#7. define a function to get the response from your agent
def get_response(query, chat_history):
    template = """
    You are an expert in so many domains, especialty good at summarizing information.
    Chat history: {chat_history}
    Here are the data need to summary: {context}
    Here is the question: {query}
    """

    prompt = ChatPromptTemplate.from_template(template)
    model = OllamaLLM(model="llama3.2")
    chain = prompt | model
    context = retriever.invoke(query)
    
    return chain.invoke({
        "chat_history": chat_history,
        "context": context,
        "query": query
    })
    
#8. build conversation flow
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("human"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("ai"):
            st.markdown(message.content)
    else:
        st.warning(f"Unsupported message type {type(message)}")
        
#9. user input query
query = st.chat_input("What do you want to know about your documents?")
if query is not None and query != "":
    st.session_state.chat_history.append(HumanMessage(query))
    
    with st.chat_message("human"):
        st.markdown(query)
    
    with st.chat_message("ai"):
        ai_response = get_response(query,st.session_state.chat_history)
        st.markdown(ai_response)
        
    st.session_state.chat_history.append(AIMessage(content=ai_response))
       