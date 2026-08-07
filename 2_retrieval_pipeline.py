from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persistent_directory = "db/chroma_db"

# Load embeddings and vector store
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Search for relevant documetns
query = "Why did Binance move its operations out of mainland China?"

retriever = db.as_retriever(search_kwargs={"k": 5})

retriever = db.as_retriever(
   search_type="similarity_score_threshold",    
   search_kwargs={
      "k": 5,
      "score_threshold": 0.3 # Only return chunks with a similarity score above 0.3
    }
)

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    
   
# Combine the query and the relevant document contents
combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I couldn't find the answer in the provided documents."
"""

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)

#Display the full result and content only
print("--- Generated Response ---")
# print("Full Result:")
# print(result)
print("\nContent Only:")
print(result.content)



# Synthetic Question Generations:

# 1. How much money did Binance raise through its initial coin offering (ICO)?
# 2. Why did Binance move its operations out of mainland China?
# 3. What specific legal type of public benefit corporation is Anthropic registered as?
# 4. What primary AI models and developer tools (such as Claude) are listed under Anthropic's product suite?
# 5. What was the name of Tesla's first production car, which used a Lotus Elise glider?
# 6. Which vehicle model completed Tesla's "Secret Master Plan" transition to a mass-market, lower-price car?
# 7. What was the name of the first privately developed liquid-fueled rocket to reach orbit?
# 8. What is the name of the satellite internet constellation operated by SpaceX to provide global broadband coverage?
# 9. What was the very first Google Doodle ever created in 1998, and what event was it for?
# 10. In what year did Google launch its free web-based email service, Gmail?


