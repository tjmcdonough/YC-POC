import os
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from typing import List, Dict, Union
from openai import OpenAI
import base64
import io
from PIL import Image
from pydantic import SecretStr

# LangChain tracing configuration
default_model = "gpt-4o-mini"
embedding_model = "text-embedding-3-small"


class LLMService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            from langchain.callbacks.manager import CallbackManager
            from langchain.callbacks.tracers import LangChainTracer

            tracer = LangChainTracer()
            callback_manager = CallbackManager([tracer])

            # Create SecretStr from API key
            api_key = SecretStr(os.environ["OPENAI_API_KEY"])

            self.llm = ChatOpenAI(temperature=0,
                                  model="gpt-4o-mini",
                                  api_key=api_key,
                                  max_tokens=4096)

            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                                               api_key=api_key)

            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=100)

            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            self._initialized = True

    def load_image_from_base64(self, image_base64: str) -> Image.Image:
        """Convert base64 string to PIL Image"""
        try:
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            return image
        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}")

    def encode_image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 string"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise ValueError(f"Failed to encode image: {str(e)}")

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks for processing"""
        return self.text_splitter.split_text(text)

    def generate_summary(self, text: str) -> str:
        """Generate a comprehensive summary of the given text"""
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""Generate a comprehensive summary of the following text. 
            Focus on key points, main ideas, and important details:

            {text}

            Summary:""")
        response = self.llm.invoke(prompt.format(text=text[:4000]))
        return str(response.content)

    def analyze_query(self, query: str) -> Dict:
        """Analyze the query and determine its type and characteristics"""
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
        if any(word in query.lower()
               for word in ["compare", "difference", "versus", "vs"]):
            query_type = "comparative"
        elif any(word in query.lower()
                 for word in ["when", "timeline", "chronological"]):
            query_type = "temporal"
        elif any(word in query.lower()
                 for word in ["sentiment", "opinion", "feel"]):
            query_type = "sentiment"
        elif any(word in query.lower()
                 for word in ["trend", "pattern", "change over time"]):
            query_type = "trend"

        return {"analysis": str(response.content), "type": query_type}

    def analyze_image(self, image_input: Union[str, Image.Image]) -> str:
        """Analyze an image using OpenAI with proper message formatting"""
        try:
            # Handle different input types
            if isinstance(image_input, str):
                if os.path.isfile(image_input):
                    image_base64 = self.encode_image_to_base64(image_input)
                else:
                    image_base64 = image_input
            elif isinstance(image_input, Image.Image):
                buffer = io.BytesIO()
                image_input.save(buffer, format="PNG")
                image_base64 = base64.b64encode(
                    buffer.getvalue()).decode('utf-8')
            else:
                raise ValueError("Invalid image input type")

            # Create ImagePromptTemplate
            message_content = [{
                "type":
                "text",
                "text":
                "Provide a detailed description of this image."
            }, {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}",
                    "detail": "high"
                }
            }]

            # Create the message and invoke
            messages = [HumanMessage(content=message_content)]
            response = self.llm.invoke(messages)

            return str(response.content
                       ) if response.content else "No description available"

        except Exception as e:
            return f"Error analyzing image: {str(e)}"

    def create_similar_queries(self, query: str, num_queries=4) -> str:
        """Generate similar queries based on a given query"""
        prompt = PromptTemplate(
            input_variables=["query", "num_queries"],
            template="""Generate similar queries based on the following query.

            Query: {query}

            Output: ({num_queries} queries)""")

        response = self.llm.invoke(
            prompt.format(query=query, num_queries=num_queries))

        return str(response.content)

    def pass_vector_results_as_context(self, vector_results: list,
                                       queries: str) -> str:
        """
        Process vector search results and a query string to create context for the LLM.

        Args:
            vector_results (list): List of document chunks from vector search
            queries (str): The user's query string

        Returns:
            str: Formatted context string for LLM prompt
        """
        # Initialize empty list to store processed contexts
        processed_documents = []

        # Process each vector result
        for doc in vector_results:
            # Format each document with its metadata
            doc_text = f"Content: {doc.page_content}"
            if hasattr(doc, 'metadata') and doc.metadata:
                doc_text += f"\nSource: {doc.metadata.get('source', 'Unknown')}"
                doc_text += f"\nPage: {doc.metadata.get('page', 'N/A')}"
            processed_documents.append(doc_text)

        # Format the final context string
        context_template = """
        Based on the following relevant information:

        Query: {query}

        {contexts}

        Please provide a comprehensive response that addresses the query while maintaining accuracy and relevance.
        """

        # Build the contexts string
        formatted_contexts = "\nRelevant documents:\n"
        for i, doc in enumerate(processed_documents, 1):
            formatted_contexts += f"\n{i}. {doc}\n"

        # Return the final formatted context
        return context_template.format(query=queries,
                                       contexts=formatted_contexts)

    def batch_process_images(self,
                             image_paths: List[str],
                             analysis_type: str = "general") -> Dict[str, str]:
        """Process multiple images and return their analyses"""
        results = {}
        for image_path in image_paths:
            try:
                results[image_path] = self.analyze_image(image_path)
            except Exception as e:
                results[image_path] = f"Error processing image: {str(e)}"
        return results

    def __str__(self):
        return "LLM Service"
