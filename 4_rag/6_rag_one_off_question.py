import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utils.fastembed import FastEmbedEmbeddings
from langchain_community.chat_models import ChatOllama

# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(
    current_dir, "db", "chroma_db_with_metadata")

# Define the embedding model
embeddings = FastEmbedEmbeddings( # type:ignore
    model_name="BAAI/bge-small-en-v1.5"
)  # Update to a valid embedding model if needed
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,
            collection_name="my_collection",
            embedding_function=embeddings)

# Define the user's question
# query = "How can I learn more about LangChain?"
query = "How did Juliet Die in the play \"Romeo and Juliet\"? Keep your answer concise and short."

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

# Combine the query and the relevant document contents
combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)

# Create a ChatOpenAI model
# model = ChatOpenAI(model="gpt-4o")
model = ChatOllama(
    model="llama3",
    base_url=os.getenv('OLLAMA_SERVER_URL', "http://localhost:11434")
)

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)
