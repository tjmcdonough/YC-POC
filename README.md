# Document Processing and Querying System

A comprehensive document processing and querying system that enables multi-format document handling, processing, and search capabilities across various file formats including PDF, Markdown, Text, HTML, and RTF files.

## Features

- **Multi-format Support**: Handle PDF, Markdown, Text, HTML, RTF files
- **Document Analysis**:
  - LLM-based summarization
  - Vector search capabilities
  - Template-based querying
  - Image analysis using gpt-4o-mini model
- **Batch Processing**:
  - Concurrent document processing
  - Progress tracking
  - Size control
- **Advanced Querying**:
  - Timeline Analysis
  - Technical Details
  - Sentiment Analysis
  - Trend Analysis
- **Robust Architecture**:
  - ThreadedConnectionPool for database operations
  - ThreadPoolExecutor for concurrent processing
  - SOLID principles implementation

## Tech Stack

- **Frontend**: Streamlit
- **Database**: PostgreSQL with ThreadedConnectionPool
- **Processing**: ThreadPoolExecutor
- **AI/ML**: 
  - LLM integration with gpt-4o-mini
  - Vector search for semantic querying
- **Document Processing**: Custom handlers for multiple formats

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - Database configuration (if using custom PostgreSQL instance):
     - `PGDATABASE`
     - `PGUSER`
     - `PGPASSWORD`
     - `PGHOST`
     - `PGPORT`

4. Run the application:
   ```bash
   streamlit run main.py
   ```

## Usage

1. **Document Upload**:
   - Select file(s) to upload
   - Choose batch size for processing
   - Monitor processing progress

2. **Query Documents**:
   - Use free-form queries or templates
   - Apply filters (date range, file type)
   - View results with relevance scores

3. **Advanced Features**:
   - Template-based analysis
   - Vector similarity search
   - Concurrent processing
   - Progress tracking

## Project Structure

```
├── components/           # UI components
├── models/              # Data models
├── services/            # Core services
│   ├── database.py      # Database operations
│   ├── file_handler.py  # File processing
│   ├── llm_service.py   # LLM integration
│   └── vector_store.py  # Vector operations
└── utils/               # Utilities
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
