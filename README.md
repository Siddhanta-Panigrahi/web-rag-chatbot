# WebRAG Chatbot 🤖

A **Retrieval-Augmented Generation (RAG) Chatbot** built with **Streamlit, FAISS, HuggingFace embeddings, and LLaMA 3.1**.  
This chatbot retrieves context from a website, indexes it using FAISS, and generates smart answers with LLaMA.  

---

## 🚀 Features
- 🔎 **Web Content Retrieval** – fetch text from a given website  
- 📂 **FAISS Vector Indexing** – efficient similarity search  
- 🧠 **HuggingFace Embeddings** – for semantic understanding  
- 💬 **LLaMA 3.1** – natural language responses  
- 🎨 **Streamlit UI** – clean chat interface with chat history  


---

## Create Virtual Environment
python -m venv virenv
- source virenv/bin/activate - #Linux/Mac
- virenv\Scripts\activate   -  #Windows


----


## Install Dependencies
pip install -r requirements.txt


----

## Add your API Key 
API_KEY="your_api_key_here"

----

## Run the App
streamlit run "your_file_name".py
