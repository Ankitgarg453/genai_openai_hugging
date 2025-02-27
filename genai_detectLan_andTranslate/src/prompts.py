from langchain.prompts import PromptTemplate
custom_prompt = PromptTemplate(
    template="You are a helpful AI. Answer this question based on retrieved document:\n{context}\n\nQuestion: {question}\n\nAnswer:",
    input_variables=["context", "question"]
)