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

# LangChain tracing configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "pr-rundown-king-67"
default_model = "gpt-4o-mini"
embedding_model = "text-embedding-3-small"

class LLMService:
    def __init__(self):
        from langchain.callbacks.manager import CallbackManager
        from langchain.callbacks.tracers import LangChainTracer

        tracer = LangChainTracer()
        callback_manager = CallbackManager([tracer])

        self.llm = ChatOpenAI(
            temperature=0,
            model=default_model,
            api_key=os.environ["OPENAI_API_KEY"],
            callbacks=callback_manager
        )

        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=os.environ["OPENAI_API_KEY"],
            callbacks=callback_manager
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
            template="""Generate a comprehensive summary of the following text. 
            Focus on key points, main ideas, and important details:

            {text}
            
            Summary:""")
        response = self.llm.invoke(prompt.format(text=text[:4000]))
        return str(response.content)

    def analyze_query(self, query: str) -> Dict:
        prompt = PromptTemplate(
            input_variables=["query"],
            template="""Analyze the following query and provide:
            1. The main intent or purpose
            2. Key concepts or entities mentioned
            3. Type of analysis requested (e.g., summary, comparison, timeline)
            4. Expected response format
            
            Query: {query}
            
            Analysis:""")
        response = self.llm.invoke(prompt.format(query=query))

        # Determine query type based on content
        query_type = "semantic"
        if any(word in query.lower() for word in ["compare", "difference", "versus", "vs"]):
            query_type = "comparative"
        elif any(word in query.lower() for word in ["when", "timeline", "chronological"]):
            query_type = "temporal"
        elif any(word in query.lower() for word in ["sentiment", "opinion", "feel"]):
            query_type = "sentiment"
        elif any(word in query.lower() for word in ["trend", "pattern", "change over time"]):
            query_type = "trend"

        return {"analysis": str(response.content), "type": query_type}

    def analyze_image(self, image_base64: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=default_model,
                messages=[{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": "Please describe the content of this image in detail."
                    }, {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }]
                }])

            return str(response.choices[0].message.content) if response.choices[0].message.content else "No description available"
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
