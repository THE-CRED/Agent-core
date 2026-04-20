"""
Structured output examples.

Demonstrates how to get typed, validated responses using Pydantic schemas.
"""

from pydantic import BaseModel, Field

from agent import Agent

# ============================================================================
# Basic Structured Output
# ============================================================================

class MovieReview(BaseModel):
    """A movie review with structured fields."""
    title: str
    rating: int = Field(ge=1, le=10, description="Rating from 1-10")
    summary: str
    pros: list[str]
    cons: list[str]


def basic_structured():
    """Get structured data from LLM."""
    agent = Agent(
        provider="openai",
        model="gpt-4o",
    )

    response = agent.json(
        "Write a review of the movie Inception",
        schema=MovieReview,
    )

    review: MovieReview = response.output
    print(f"Title: {review.title}")
    print(f"Rating: {review.rating}/10")
    print(f"Summary: {review.summary}")
    print(f"Pros: {', '.join(review.pros)}")
    print(f"Cons: {', '.join(review.cons)}")


# ============================================================================
# Nested Structures
# ============================================================================

class Address(BaseModel):
    """A physical address."""
    street: str
    city: str
    country: str
    postal_code: str


class Person(BaseModel):
    """A person with contact information."""
    name: str
    age: int
    email: str
    address: Address


def nested_structures():
    """Extract nested structured data."""
    agent = Agent(
        provider="anthropic",
        model="claude-sonnet",
    )

    response = agent.json(
        "Generate a fictional person's profile with full contact details",
        schema=Person,
    )

    person: Person = response.output
    print(f"Name: {person.name}")
    print(f"Age: {person.age}")
    print(f"Email: {person.email}")
    print(f"Address: {person.address.street}, {person.address.city}")


# ============================================================================
# Data Extraction
# ============================================================================

class ExtractedEntities(BaseModel):
    """Entities extracted from text."""
    people: list[str] = Field(description="Names of people mentioned")
    organizations: list[str] = Field(description="Organizations mentioned")
    locations: list[str] = Field(description="Locations mentioned")
    dates: list[str] = Field(description="Dates mentioned")


def entity_extraction():
    """Extract entities from unstructured text."""
    agent = Agent(
        provider="openai",
        model="gpt-4o",
    )

    text = """
    On January 15, 2024, Apple CEO Tim Cook announced a partnership with
    Microsoft at their headquarters in Cupertino, California. The deal was
    also attended by Satya Nadella and was later discussed at a conference
    in New York City on February 1st.
    """

    response = agent.json(
        f"Extract all entities from this text:\n\n{text}",
        schema=ExtractedEntities,
    )

    entities: ExtractedEntities = response.output
    print(f"People: {entities.people}")
    print(f"Organizations: {entities.organizations}")
    print(f"Locations: {entities.locations}")
    print(f"Dates: {entities.dates}")


# ============================================================================
# Classification
# ============================================================================

class SentimentAnalysis(BaseModel):
    """Sentiment analysis result."""
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(ge=0, le=1, description="Confidence score 0-1")
    key_phrases: list[str] = Field(description="Key phrases that indicate sentiment")


def sentiment_classification():
    """Classify sentiment with structured output."""
    agent = Agent(
        provider="openai",
        model="gpt-4o",
    )

    texts = [
        "This product is absolutely amazing! Best purchase ever!",
        "The service was okay, nothing special but not bad either.",
        "Terrible experience. Would not recommend to anyone.",
    ]

    for text in texts:
        response = agent.json(
            f"Analyze the sentiment of this text: {text}",
            schema=SentimentAnalysis,
        )

        result: SentimentAnalysis = response.output
        print(f"Text: {text[:50]}...")
        print(f"  Sentiment: {result.sentiment} (confidence: {result.confidence:.2f})")
        print(f"  Key phrases: {result.key_phrases}")
        print()


# ============================================================================
# Code Analysis
# ============================================================================

class CodeReview(BaseModel):
    """Code review result."""
    issues: list[str] = Field(description="List of issues found")
    suggestions: list[str] = Field(description="Improvement suggestions")
    complexity_score: int = Field(ge=1, le=10, description="Complexity 1-10")
    security_concerns: list[str] = Field(description="Security issues if any")


def code_analysis():
    """Analyze code with structured feedback."""
    agent = Agent(
        provider="anthropic",
        model="claude-sonnet",
    )

    code = '''
def process_user_input(user_data):
    query = f"SELECT * FROM users WHERE id = {user_data['id']}"
    result = db.execute(query)
    return eval(result[0]['data'])
    '''

    response = agent.json(
        f"Review this Python code for issues:\n\n```python\n{code}\n```",
        schema=CodeReview,
    )

    review: CodeReview = response.output
    print("Code Review Results:")
    print(f"Complexity: {review.complexity_score}/10")
    print(f"Issues: {review.issues}")
    print(f"Security concerns: {review.security_concerns}")
    print(f"Suggestions: {review.suggestions}")


if __name__ == "__main__":
    # Uncomment to run:
    # basic_structured()
    # nested_structures()
    # entity_extraction()
    # sentiment_classification()
    # code_analysis()

    print("Uncomment an example function to run it!")
