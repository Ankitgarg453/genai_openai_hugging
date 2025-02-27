from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as PineconeLangChain
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceEndpoint
from pinecone import Pinecone
from langchain_community.llms import OpenAI
from genai_detectLan_andTranslate.src.prompts import custom_prompt

def initialize_pinecone(pinecone_key, index_name, dimension=384):
    """Initialize Pinecone index"""
    pc = Pinecone(api_key=pinecone_key)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine"
        )
    return pc.Index(index_name)

async def process_pdf_and_store(pinecone_key, index_name, text_chunks):
    """Convert PDF text to vectors and store in Pinecone"""
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector = embedding_model.embed_query("Hello world")
    print(len(vector)) 
    # print(embedding_model)
    pine_index = initialize_pinecone(pinecone_key, index_name)
    
    vectors_to_store = [
        (f"doc_{i}", embedding_model.embed_query(text.page_content), {"text": text.page_content})
        for i, text in enumerate(text_chunks)
    ]
    
    pine_index.upsert(vectors_to_store)
    print(f"Stored {len(vectors_to_store)} embeddings in Pinecone.")
    # return pine_index

async def retrieve_answer(openapi_token, pinecone_key, index_name):
    """Retrieve answers using RetrievalQA"""
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    pine_index = initialize_pinecone(pinecone_key, index_name)
    vectorstore = PineconeLangChain(
        index=pine_index,
        embedding=embedding_model,
        text_key="text"
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    # llm = HuggingFaceEndpoint(
    #     repo_id="facebook/bart-large-cnn",
    #     huggingfacehub_api_token=huggingapi_token,
    #     temperature=0.3,
    #     model_kwargs={"max_length": 256}
    # )
    llm = OpenAI(openai_api_key=openapi_token)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff",chain_type_kwargs={"prompt": custom_prompt})
    # result = qa_chain.run(query_text)
    # print(f"Answer: {result}")
    return qa_chain