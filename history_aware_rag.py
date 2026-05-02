from langchain_core.tools import retriever
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import os

load_dotenv()

#-------------------------------------------------------------------
#                       Set up Groq & Chroma
#-------------------------------------------------------------------
persist_directory = "db/cdb"
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)

#-------------------------------------------------------------------
#                      History aware section
#-------------------------------------------------------------------

#list holding previous chat messages
chat_history = []


#Rewriting user question to be standalone searchable
def ask_rag(query:str):
    """Ask RAG with chat history"""
    if chat_history:
        messages = [
            SystemMessage( content = "You are a legal document assistant, rewrite the new question to be standalone and searchable. Just return the new question" ),
            ] + chat_history + [
                HumanMessage(content= f"New Question: {query}")
            ] 

        result = model.invoke(messages)
        search_query = result.content.strip()
        print(f"Searching for: {search_query}")
    else:
        search_query = query

    retriever = db.as_retriever(
        search_kwargs={"k": 6}
    )
    docs = retriever.invoke(search_query)

    # Combine retrieved documents into context
    context = "\n\n".join([f"Document:\n{doc.page_content}" for doc in docs])
    
    # Build the prompt for the model to answer the question
    answer_messages = [
        SystemMessage(content=(
            "You are an expert legal analyst reviewing contracts on behalf of the user.\n"
            "Your job is to:\n"
            "- Identify and explain any clauses that could be risky, unfair, or dangerous to the Borrower.\n"
            "- Answer questions using the provided contract excerpts.\n"
            "- If a clause sounds legal but is actually harmful, flag it and explain why.\n"
            "- Tell the term number and name if available.\n"
            "- Do not use ** in answers.\n"
            "- If the answer is not in the context, say: 'This information is not available in the provided documents.'\n\n"
            "Contract context:\n" + context
        )),
        HumanMessage(content=search_query)
    ]
    
    # Get the answer and print it
    print("\nThinking...")
    answer_result = model.invoke(answer_messages)
    answer = answer_result.content
    print(f"\n---- Answer ----\n{answer}\n")
    
    # Save to chat history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=answer))

#starting the chat
def start_chat():
    print("Welcome to legal RAG chatbot. Type 'exit' to quit.\n")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        else:
            ask_rag(query)


if __name__ == "__main__":
    start_chat()