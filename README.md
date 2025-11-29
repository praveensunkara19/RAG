# RAG - Retrieval-Augmented Generation Application

* Persionalized RAG for query retrievals.
* Quick Q/A with the given knowledge bases.

---

# Demo Images:
<p align="center">
  <img src="images/demo.png" alt="demo image" width="600"/>
</p>


## Tech stack

Libraries  used to get done the project:
```
PDFLoader - to purse the data
TextSplit(RecursiveCharacterTextSplitter) - to split the text into chunks
sentence-transformers(all-mpnet-base-v2) - Embedding the chunks
ChromaDB - to store the Embedding Vectors
Groq-LLM(openai/gpt-oss-120b) - to summarize the context along with the prompt
```
---

# Workflow of the RAG Pipeline
<p align="center">
  <img src="images/rag_pipeline.jpg" alt="demo image" width="600"/>
  <img src="images/rag_chroma.png" alt="demo image" width="600"/>
</p>

# Project setup Locally:
Go to Terminal or bash of you project folder:

```
git clone https://github.com/praveensunkara19/RAG.git

cd RAG

python -m venv myenv 

myenv/scripts/activate

pip install -r requirements.txt

streamlit run app.py
```

#Future scope:
* Adding persionlization like speech
* Retrieval of images - videos 
* Industry level optimization for better retrieval with - FAISS, PineCone
* Works with all kind of document types.


---------------------  *****"References"******* ----------------


1) https://huggingface.co/MBZUAI/LaMini-T5-738M   LLM to process the chunks of the vectorised db to give the results
2) https://www.trychroma.com/  for chromadb and related libraries.
3) https://python.langchain.com/v0.2/docs/integrations/platforms/huggingface/   for HuggingFace LLMs and Endpoints
4) https://pypi.org/project/transformers/   used to train the model
5) https://pypi.org/project/sentence-transformers/   for the SentenceTransformerEmbeddings used as the text_spitting
6) https://docs.streamlit.io/   used to display the output (UI)
7) https://fastapi.tiangolo.com/  it served as the API for the Document Retreaval System wiht user_key methods
8) https://www.sqlite.org/   to access and organize the user data such as responses ,frequency etc.,,
