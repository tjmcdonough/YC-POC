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
from pydantic import SecretStr

# LangChain tracing configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "pr-rundown-king-67"

class LLMService:
    def __init__(self):
        from langchain.callbacks.manager import CallbackManager
        from langchain.callbacks.tracers import LangChainTracer
        
        tracer = LangChainTracer()
        callback_manager = CallbackManager([tracer])
        
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4-turbo-preview",
            api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
            callbacks=callback_manager
        )
        
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=SecretStr(os.environ["OPENAI_API_KEY"])
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def split_text(self, text: str) -> List[str]:
        return self.text_splitter.split_text(text)

    def generate_summary(self, text: str) -> str:
        prompt = PromptTemplate(
            input_variables=["text"],
            template="Summarize the following text:\n\n{text}"
        )
        response = self.llm.invoke(prompt.format(text=text[:4000]))
        return str(response.content)

    def analyze_query(self, query: str) -> Dict:
        prompt = PromptTemplate(
            input_variables=["query"],
            template="Analyze the following query and classify its intent and type:\n\n{query}"
        )
        response = self.llm.invoke(prompt.format(query=query))
        return {
            "analysis": str(response.content),
            "type": "semantic" if "why" in query.lower() or "how" in query.lower() else "factual"
        }

    def analyze_image(self, image_base64: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please describe the content of this image in detail."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }],
                max_tokens=300
            )
            
            return str(response.choices[0].message.content) if response.choices[0].message.content else "No description available"
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
