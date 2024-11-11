from abc import ABC, abstractmethod
import zipfile
import io
from typing import BinaryIO, List, Dict, Tuple
import docx
import fitz  # PyMuPDF
from PIL import Image
import json
import xml.etree.ElementTree as ET
import csv
import pandas as pd
import markdown
from bs4 import BeautifulSoup
import re
import base64

class FileHandler(ABC):
    @abstractmethod
    def extract_text(self, file: BinaryIO) -> str:
        pass

    def _get_image_summary(self, image: Image.Image, llm_service) -> str:
        # Convert image to base64 for API
        buffered = io.BytesIO()
        image.save(buffered, format=image.format or 'PNG')
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return llm_service.analyze_image(img_str)

class PDFHandler(FileHandler):
    def extract_text(self, file: BinaryIO, llm_service=None) -> str:
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        image_summaries = []
        
        for page in pdf_document:
            text += page.get_text()
            if llm_service:  # Only process images if LLM service is provided
                # Extract images
                images = page.get_images()
                for img_index, img in enumerate(images):
                    try:
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_data = base_image["image"]
                        # Convert to PIL Image
                        image = Image.open(io.BytesIO(image_data))
                        # Get image summary from LLM
                        image_summary = self._get_image_summary(image, llm_service)
                        image_summaries.append(f"Image {img_index + 1}: {image_summary}")
                    except Exception as e:
                        print(f"Error processing image {img_index + 1}: {str(e)}")
                        continue
        
        # Combine text and image summaries
        if image_summaries:
            combined_text = text + "\n\nImage Descriptions:\n" + "\n".join(image_summaries)
            return combined_text
        return text

class DocxHandler(FileHandler):
    def extract_text(self, file: BinaryIO, llm_service=None) -> str:
        doc = docx.Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        image_summaries = []
        
        if llm_service:  # Only process images if LLM service is provided
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    try:
                        # Extract image
                        image_data = rel.target_part.blob
                        image = Image.open(io.BytesIO(image_data))
                        # Get image summary
                        image_summary = self._get_image_summary(image, llm_service)
                        image_summaries.append(image_summary)
                    except Exception as e:
                        print(f"Error processing image in DOCX: {str(e)}")
                        continue
        
        # Combine text and image summaries
        if image_summaries:
            combined_text = text + "\n\nImage Descriptions:\n" + "\n".join(image_summaries)
            return combined_text
        return text

class ImageHandler(FileHandler):
    def extract_text(self, file: BinaryIO, llm_service=None) -> str:
        try:
            image = Image.open(file)
            if llm_service:
                return self._get_image_summary(image, llm_service)
            return f"Image dimensions: {image.size}, format: {image.format}"
        except Exception as e:
            return f"Error processing image: {str(e)}"

class JSONHandler(FileHandler):
    def extract_text(self, file: BinaryIO, llm_service=None) -> str:
        return json.dumps(json.load(file), indent=2)

class XMLHandler(FileHandler):
    def extract_text(self, file: BinaryIO, llm_service=None) -> str:
        tree = ET.parse(file)
        return ET.tostring(tree.getroot(), encoding='unicode', method='xml')

class CSVHandler(FileHandler):
    def extract_text(self, file: BinaryIO, llm_service=None) -> str:
        df = pd.read_csv(file)
        return df.to_string()

class MarkdownHandler(FileHandler):
    def extract_text(self, file: BinaryIO, llm_service=None) -> str:
        content = file.read().decode('utf-8')
        # Convert markdown to HTML
        html = markdown.markdown(content)
        # Remove HTML tags to get clean text
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text(separator='\n\n')

class TextHandler(FileHandler):
    def extract_text(self, file: BinaryIO, llm_service=None) -> str:
        return file.read().decode('utf-8')

class HTMLHandler(FileHandler):
    def extract_text(self, file: BinaryIO, llm_service=None) -> str:
        content = file.read().decode('utf-8')
        soup = BeautifulSoup(content, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        # Get text
        text = soup.get_text(separator='\n\n')
        # Remove extra whitespace and empty lines
        text = re.sub(r'\n\s*\n', '\n\n', text.strip())
        return text

class RTFHandler(FileHandler):
    def extract_text(self, file: BinaryIO, llm_service=None) -> str:
        # Simple RTF text extraction
        content = file.read().decode('utf-8', errors='ignore')
        # Remove RTF formatting
        text = re.sub(r'[\\\{\}]|\\\w+|\{.*?\}', '', content)
        return text.strip()

class FileHandlerFactory:
    _handlers = {
        'pdf': PDFHandler(),
        'docx': DocxHandler(),
        'jpg': ImageHandler(),
        'jpeg': ImageHandler(),
        'png': ImageHandler(),
        'json': JSONHandler(),
        'xml': XMLHandler(),
        'csv': CSVHandler(),
        'md': MarkdownHandler(),
        'txt': TextHandler(),
        'html': HTMLHandler(),
        'htm': HTMLHandler(),
        'rtf': RTFHandler()
    }

    @classmethod
    def get_handler(cls, file_type: str) -> FileHandler:
        handler = cls._handlers.get(file_type.lower())
        if not handler:
            raise ValueError(f"Unsupported file type: {file_type}")
        return handler
