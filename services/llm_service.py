import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from typing import List, Dict
from openai import OpenAI
import base64
import io
from PIL import Image

class LLMService:
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo",
            api_key=os.environ["OPENAI_API_KEY"]
        )
        self.embeddings = OpenAIEmbeddings(
            api_key=os.environ["OPENAI_API_KEY"]
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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

    def analyze_image(self, image_base64: str) -> str:
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please describe the content of this image in detail."
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    ]
                }
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=messages,
                max_tokens=300
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
