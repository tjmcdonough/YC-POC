import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from typing import List, Dict, Union, Optional
from openai import OpenAI
import base64
import io
from PIL import Image
import imghdr

# LangChain tracing configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "pr-rundown-king-67"

default_model = "gpt-4-vision-preview"  # Updated to use vision model
embedding_model = "text-embedding-3-small"

class LLMService:
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4",  # Regular model for text processing
            api_key=os.environ["OPENAI_API_KEY"],
            max_tokens=4096
        )

        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=os.environ["OPENAI_API_KEY"]
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def validate_image(self, image: Union[str, bytes, Image.Image]) -> Image.Image:
        """Validate and convert input to PIL Image"""
        try:
            if isinstance(image, str):
                # Handle base64 string
                if ',' in image:
                    image = image.split(',')[1]
                try:
                    image_data = base64.b64decode(image)
                    img = Image.open(io.BytesIO(image_data))
                except:
                    # Handle file path
                    if os.path.isfile(image):
                        img = Image.open(image)
                    else:
                        raise ValueError("Invalid image string format")
            elif isinstance(image, bytes):
                img = Image.open(io.BytesIO(image))
            elif isinstance(image, Image.Image):
                img = image
            else:
                raise ValueError("Unsupported image type")

            # Validate image format
            if img.format not in ['JPEG', 'PNG', 'WEBP']:
                img = img.convert('RGB')
                # Convert to PNG for consistency
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                img = Image.open(buffer)

            return img
        except Exception as e:
            raise ValueError(f"Image validation failed: {str(e)}")

    def encode_image(self, image: Image.Image, format: str = 'PNG') -> str:
        """Encode PIL Image to base64 string"""
        try:
            buffer = io.BytesIO()
            image.save(buffer, format=format)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            raise ValueError(f"Image encoding failed: {str(e)}")

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

            Summary:"""
        )
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

            Analysis:"""
        )
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

    def analyze_image(self, image_input: Union[str, bytes, Image.Image], analysis_type: str = "general") -> str:
        """Analyze an image using OpenAI's GPT-4 Vision model"""
        try:
            # Validate and prepare image
            img = self.validate_image(image_input)
            image_base64 = self.encode_image(img)

            # Use the OpenAI client with GPT-4 Vision
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an expert image analyst. Analyze the image according to these guidelines: Be precise and factual, focus on {analysis_type} aspects."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please analyze this image in detail."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            # Extract and return the analysis
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content
            return "No analysis generated"
            
        except Exception as e:
            error_msg = f"Analysis error: {str(e)}"
            print(f"Image analysis failed: {error_msg}")  # Log the error
            return error_msg

    def batch_analyze_images(
        self,
        images: List[Union[str, bytes, Image.Image]],
        analysis_type: str = "general"
    ) -> Dict[int, str]:
        """Batch process multiple images and return their analyses"""
        results = {}
        for idx, image in enumerate(images):
            try:
                results[idx] = self.analyze_image(image, analysis_type)
            except Exception as e:
                error_msg = f"Error processing image {idx}: {str(e)}"
                print(f"Batch processing error: {error_msg}")  # Log the error
                results[idx] = error_msg
        return results
