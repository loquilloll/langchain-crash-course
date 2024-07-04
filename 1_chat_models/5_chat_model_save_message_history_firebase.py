# Example Source: https://python.langchain.com/v0.2/docs/integrations/memory/google_firestore/

from dotenv import load_dotenv
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core import chat_history
from langchain_community.chat_models import ChatOllama


load_dotenv()

# Setup Firebase Firestore
PROJECT_ID = "langchain-demo-abf48"
SESSION_ID = "user_session_new"  # This could be a username or a unique ID
# COLLECTION_NAME = "chat_history"
CONNECTION_STRING='sqlite:///chat_history.db'

# Initialize SQLlite Chat Message History
print("Initializing SQLlite Chat Message History...")
chat_history = SQLChatMessageHistory(session_id=SESSION_ID, connection=CONNECTION_STRING)
print("Chat History Initialized.")
print("Current Chat History:", chat_history.messages)

# Create a ChatOllama model
model = ChatOllama(model="llama3")

print("Start chatting with the AI. Type 'exit' to quit.")

while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break

    chat_history.add_user_message(human_input)

    ai_response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content) # type:ignore

    print(f"AI: {ai_response.content}")
