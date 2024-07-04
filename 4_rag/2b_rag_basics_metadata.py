import os

from langchain_community.vectorstores import Chroma
from utils.fastembed import FastEmbedEmbeddings
from chromadb.utils.embedding_functions import create_langchain_embedding

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# Define the embedding model
embeddings = FastEmbedEmbeddings( # type:ignore
    model_name="BAAI/bge-small-en-v1.5"
)  # Update to a valid embedding model if needed
embedding_function = create_langchain_embedding(embeddings)

# Load the existing vector store with the embedding function
db = Chroma(
    persist_directory=persistent_directory,
    collection_name="my_collection",
    embedding_function=embedding_function
)

# Define the user's question
query = "How did Juliet die?"
# docs = db.similarity_search(query)
# print(docs)

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.1},
)
relevant_docs = retriever.invoke(query)
# print(relevant_docs[0])

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    print(f"Source: {doc.metadata['source']}\n")
