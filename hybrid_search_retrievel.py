from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

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


#-----------------------------------------------------------
#                     Vector Search
#-----------------------------------------------------------

vector_retriever = db.as_retriever(
    search_kwargs = {"k":7}
)

# test = vector_retriever.invoke("what does this document is for?")
# print("Vector Search Results:")
# print("\n---------1-----------")
# print(test[0].page_content)
# print("\n---------2-----------")
# print(test[1].page_content)
# print("\n---------3-----------")
# print(test[2].page_content)



#-----------------------------------------------------------
#                     BM25 Search
#-----------------------------------------------------------

stored = db.get(include=["documents", "metadatas"])
bm25_retriever = BM25Retriever.from_texts(
    texts = stored["documents"],
    metadatas= stored["metadatas"]
)
bm25_retriever.k=7

# bm25_results = bm25_retriever.invoke("contract")

# print("\nBM25 Search Results")
# print("\n---------1-----------")
# print(bm25_results[0].page_content)
# print("\n---------2-----------")
# print(bm25_results[1].page_content)
# print("\n---------3-----------")
# print(bm25_results[2].page_content)

#-----------------------------------------------------------
#                     Ensemble Search
#-----------------------------------------------------------

ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever,bm25_retriever],
    weights=[0.7,0.3]
)

# ensemble_results = ensemble_retriever.invoke("contract")

# print("\nEnsemble Search Results")
# print("\n---------1-----------")
# print(ensemble_results[0].page_content)
# print("\n---------2-----------")
# print(ensemble_results[1].page_content)
# print("\n---------3-----------")
# print(ensemble_results[2].page_content)


#-----------------------------------------------------------
#                     Answer Generation
#-----------------------------------------------------------

query = input("Enter your Query: ")

combined_input = f"""
Based on the retrieved documents, answer this query: {query}

Documents: 
{ensemble_retriever.invoke(query)}

If the answer is not present in the retrieved documents, reply with 'This information is not available in the provided documents.'

"""
messages = [
    SystemMessage(content=("You are an expert legal analyst reviewing contracts on behalf of the borrower. Your job is to identify and explain any clauses that could be risky, unfair, or dangerous to the borrower.")),
    HumanMessage(content=combined_input)
]

result = model.invoke(messages)
print(result.content)