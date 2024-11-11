from abc import ABC, abstractmethod
import zipfile
import io
from typing import BinaryIO, List, Dict
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

class FileHandler(ABC):
    @abstractmethod
    def extract_text(self, file: BinaryIO) -> str:
        pass

class PDFHandler(FileHandler):
    def extract_text(self, file: BinaryIO) -> str:
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in pdf_document:
            text += page.get_text()
        return text

class DocxHandler(FileHandler):
    def extract_text(self, file: BinaryIO) -> str:
        doc = docx.Document(file)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])

class ImageHandler(FileHandler):
    def extract_text(self, file: BinaryIO) -> str:
        image = Image.open(file)
        return f"Image dimensions: {image.size}, format: {image.format}"

class JSONHandler(FileHandler):
    def extract_text(self, file: BinaryIO) -> str:
        return json.dumps(json.load(file), indent=2)

class XMLHandler(FileHandler):
    def extract_text(self, file: BinaryIO) -> str:
        tree = ET.parse(file)
        return ET.tostring(tree.getroot(), encoding='unicode', method='xml')

class CSVHandler(FileHandler):
    def extract_text(self, file: BinaryIO) -> str:
        df = pd.read_csv(file)
        return df.to_string()

class MarkdownHandler(FileHandler):
    def extract_text(self, file: BinaryIO) -> str:
        content = file.read().decode('utf-8')
        # Convert markdown to HTML
        html = markdown.markdown(content)
        # Remove HTML tags to get clean text
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text(separator='\n\n')

class TextHandler(FileHandler):
    def extract_text(self, file: BinaryIO) -> str:
        return file.read().decode('utf-8')

class HTMLHandler(FileHandler):
    def extract_text(self, file: BinaryIO) -> str:
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
    def extract_text(self, file: BinaryIO) -> str:
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
