from typing import Dict, List

class QueryTemplate:
    def __init__(self, name: str, template: str, description: str, parameters: List[str]):
        self.name = name
        self.template = template
        self.description = description
        self.parameters = parameters

    def format_query(self, params: Dict[str, str]) -> str:
        return self.template.format(**params)

QUERY_TEMPLATES = {
    'summary': QueryTemplate(
        name="Document Summary",
        template="Provide a comprehensive summary of {topic} from the documents",
        description="Get a detailed summary about a specific topic",
        parameters=['topic']
    ),
    'comparison': QueryTemplate(
        name="Compare Documents",
        template="Compare and contrast how {aspect} is discussed in documents from {date_range}",
        description="Compare how a specific aspect is covered across different documents",
        parameters=['aspect', 'date_range']
    ),
    'key_insights': QueryTemplate(
        name="Key Insights",
        template="What are the key insights about {subject} mentioned in the {file_type} documents?",
        description="Extract key insights about a specific subject",
        parameters=['subject', 'file_type']
    ),
    'timeline': QueryTemplate(
        name="Timeline Analysis",
        template="Create a timeline of events related to {event} from the documents within {date_range}",
        description="Generate a chronological analysis of events",
        parameters=['event', 'date_range']
    ),
    'technical_details': QueryTemplate(
        name="Technical Details",
        template="Extract technical specifications and details about {technology} from {file_type} documents",
        description="Get detailed technical information about a specific topic",
        parameters=['technology', 'file_type']
    ),
    'sentiment_analysis': QueryTemplate(
        name="Sentiment Analysis",
        template="Analyze the sentiment and opinions about {topic} in documents from {date_range}",
        description="Understand the overall sentiment and opinions on a specific topic",
        parameters=['topic', 'date_range']
    ),
    'trend_analysis': QueryTemplate(
        name="Trend Analysis",
        template="Identify and analyze trends related to {subject} in {file_type} documents from {date_range}",
        description="Track and analyze trends over time",
        parameters=['subject', 'file_type', 'date_range']
    ),
    'recommendations': QueryTemplate(
        name="Recommendations",
        template="What recommendations are given regarding {topic} in the documents?",
        description="Extract recommendations and suggested actions",
        parameters=['topic']
    )
}
