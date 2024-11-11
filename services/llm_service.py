import os
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from typing import List, Dict

class LLMService:
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            openai_api_key=os.environ["OPENAI_API_KEY"]
        )
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.environ["OPENAI_API_KEY"]
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def split_text(self, text: str) -> List[str]:
        return self.text_splitter.split_text(text)

    def generate_summary(self, text: str) -> str:
        prompt = PromptTemplate(
            input_variables=["text"],
            template="Summarize the following text:\n\n{text}"
        )
        
        response = self.llm.predict(prompt.format(text=text[:4000]))
        return response

    def analyze_query(self, query: str) -> Dict:
        prompt = PromptTemplate(
            input_variables=["query"],
            template="Analyze the following query and classify its intent and type:\n\n{query}"
        )
        
        response = self.llm.predict(prompt.format(query=query))
        return {
            "analysis": response,
            "type": "semantic" if "why" in query.lower() or "how" in query.lower() else "factual"
        }
