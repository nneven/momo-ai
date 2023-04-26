import os
import logging
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import GoogleDriveLoader
from langchain.memory import ConversationBufferMemory

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] [%(name)s] %(message)s",
)

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "sk-xxx"

# Set up Google Drive Loader
loader = GoogleDriveLoader(
    credentials_path="credentials.json",
    token_path="token.json",
    folder_id="1Y5XUpi5egCHX4Rg5pSAqRB0v8LoTm-YP",  # ENST 100
    recursive=False  # Configure whether to recursively fetch files from subfolders
)

# Load documents
logging.info("Loading documents...")
docs = loader.load()

# Split documents into chunks
logging.info("Splitting documents...")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(docs)

# Generate embeddings and create vectorstore
logging.info("Generating embeddings...")
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

# Create retrieval chain
logging.info("Creating retrieval chain...")
# model = OpenAI()
model = ChatOpenAI(model='gpt-3.5-turbo')
retriever = vectorstore.as_retriever()
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(model, retriever)

# Start chat loop
chat_history = []
while True:
    query = input("Enter your question (or 'exit'): ")
    if query.lower() == "exit":
        break
    
    result = qa({"question": query, "chat_history": chat_history})
    print("Answer:", result["answer"])
    chat_history.append((query, result["answer"]))
