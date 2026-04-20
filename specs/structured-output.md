# Structured Output

Get typed, validated responses from LLMs using Pydantic schemas. Agent handles schema conversion, prompt engineering, and output validation automatically.

## Overview

Instead of parsing free-text responses, define a schema and get structured data:

```python
from pydantic import BaseModel
from agent import Agent

class MovieReview(BaseModel):
    title: str
    rating: int
    summary: str

agent = Agent(provider="openai", model="gpt-4o")
response = agent.json("Review the movie Inception", schema=MovieReview)

review: MovieReview = response.output
print(f"{review.title}: {review.rating}/10")
```

## Quick Start

### 1. Define a Schema

```python
from pydantic import BaseModel, Field

class Analysis(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(ge=0, le=1, description="Confidence score")
    keywords: list[str] = Field(description="Key terms found")
```

### 2. Use with Agent

```python
response = agent.json(
    "Analyze: This product is amazing!",
    schema=Analysis,
)

result: Analysis = response.output
print(result.sentiment)    # "positive"
print(result.confidence)   # 0.95
print(result.keywords)     # ["amazing", "product"]
```

## Schema Definition

### Basic Types

```python
class Person(BaseModel):
    name: str
    age: int
    height: float
    is_active: bool
```

### Optional Fields

```python
class Profile(BaseModel):
    name: str
    bio: str | None = None  # Optional
    website: str = ""       # Has default
```

### Lists and Nested Objects

```python
class Address(BaseModel):
    street: str
    city: str
    country: str

class Company(BaseModel):
    name: str
    employees: list[str]
    headquarters: Address
    subsidiaries: list[Address] = []
```

### Field Constraints

```python
from pydantic import Field

class Product(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    price: float = Field(gt=0, description="Price in USD")
    quantity: int = Field(ge=0, le=1000)
    category: str = Field(pattern=r"^[a-z]+$")
```

### Enums and Literals

```python
from enum import Enum
from typing import Literal

class Status(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"

class Request(BaseModel):
    status: Status
    priority: Literal["low", "medium", "high"]
```

### Descriptions for Better Results

Add descriptions to help the LLM understand what you want:

```python
class CodeReview(BaseModel):
    """Review of code quality and issues."""
    
    issues: list[str] = Field(
        description="List of specific issues found in the code"
    )
    severity: str = Field(
        description="Overall severity: low, medium, high, or critical"
    )
    suggestions: list[str] = Field(
        description="Actionable suggestions to improve the code"
    )
    score: int = Field(
        ge=1, le=10,
        description="Quality score from 1 (poor) to 10 (excellent)"
    )
```

## Using json()

### Basic Usage

```python
response = agent.json(
    "Extract the main points from this article...",
    schema=Summary,
)
```

### With System Prompt

```python
response = agent.json(
    "Analyze this customer feedback",
    schema=FeedbackAnalysis,
    system="You are an expert customer feedback analyst.",
)
```

### In Sessions

```python
session = agent.session()
session.run("Here's a document about climate change...")

response = session.json(
    "Extract the key statistics",
    schema=Statistics,
)
```

### Async

```python
response = await agent.json_async(
    "Parse this data",
    schema=ParsedData,
)
```

## How It Works

### Native Schema Support

Some providers (OpenAI, Gemini) support native JSON schema enforcement:

```python
# OpenAI uses response_format with json_schema
{
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "Analysis",
            "schema": {...},
            "strict": True
        }
    }
}
```

### Prompt-Based Fallback

For providers without native support (Anthropic), Agent adds schema instructions to the prompt:

```
Respond with a JSON object matching this schema:

{
  "type": "object",
  "properties": {
    "sentiment": {"type": "string"},
    "confidence": {"type": "number"}
  },
  "required": ["sentiment", "confidence"]
}

IMPORTANT: Return ONLY the JSON object, no other text.
```

### Validation and Repair

Agent validates output and attempts repair if needed:

1. Parse response as JSON
2. Extract JSON from markdown code blocks if present
3. Validate against Pydantic schema
4. Attempt JSON repair if malformed (fix brackets, trailing commas)
5. Retry validation
6. Raise `SchemaValidationError` if all attempts fail

## Error Handling

### Schema Validation Errors

```python
from agent import SchemaValidationError

try:
    response = agent.json("...", schema=MySchema)
except SchemaValidationError as e:
    print(f"Validation failed: {e.message}")
    print(f"Invalid output: {e.output}")
    print(f"Schema: {e.schema}")
```

### Graceful Fallback

```python
response = agent.json("...", schema=MySchema)

if response.output is None:
    # Validation failed but we got text
    print(f"Couldn't parse, raw text: {response.text}")
else:
    # Successfully parsed
    result: MySchema = response.output
```

## Common Patterns

### Entity Extraction

```python
class Entities(BaseModel):
    people: list[str] = Field(description="Names of people mentioned")
    organizations: list[str] = Field(description="Organization names")
    locations: list[str] = Field(description="Place names")
    dates: list[str] = Field(description="Dates in ISO format")

response = agent.json(
    f"Extract entities from: {document}",
    schema=Entities,
)
```

### Classification

```python
class Classification(BaseModel):
    category: Literal["bug", "feature", "question", "other"]
    confidence: float
    reasoning: str

response = agent.json(
    f"Classify this support ticket: {ticket}",
    schema=Classification,
)
```

### Data Transformation

```python
class StructuredResume(BaseModel):
    name: str
    email: str
    skills: list[str]
    experience: list[dict]
    education: list[dict]

response = agent.json(
    f"Parse this resume into structured data:\n{resume_text}",
    schema=StructuredResume,
)
```

### Multi-Step Analysis

```python
class Step(BaseModel):
    action: str
    reasoning: str
    result: str

class Analysis(BaseModel):
    steps: list[Step]
    conclusion: str
    confidence: float

response = agent.json(
    "Analyze this problem step by step: ...",
    schema=Analysis,
)
```

### API Response Parsing

```python
class APIResponse(BaseModel):
    success: bool
    data: dict | None = None
    error: str | None = None

response = agent.json(
    f"Parse this API response and normalize it: {raw_response}",
    schema=APIResponse,
)
```

## Best Practices

### 1. Use Descriptive Field Names

```python
# Good
class Order(BaseModel):
    customer_name: str
    order_total: float
    shipping_address: str

# Bad
class Order(BaseModel):
    n: str
    t: float
    a: str
```

### 2. Add Field Descriptions

```python
class Review(BaseModel):
    rating: int = Field(
        ge=1, le=5,
        description="Star rating from 1 (worst) to 5 (best)"
    )
```

### 3. Use Constraints

```python
class Product(BaseModel):
    price: float = Field(gt=0)  # Must be positive
    sku: str = Field(pattern=r"^[A-Z]{3}-\d{4}$")  # Format: ABC-1234
```

### 4. Keep Schemas Focused

```python
# Good - focused schema
class SentimentResult(BaseModel):
    sentiment: str
    score: float

# Bad - trying to do too much
class EverythingResult(BaseModel):
    sentiment: str
    entities: list
    summary: str
    translation: str
    keywords: list
    # ... too many fields
```

### 5. Test with Examples

```python
def test_schema_extraction():
    agent, provider = create_test_agent()
    provider.set_response(FakeResponse(
        text='{"name": "Test", "value": 42}'
    ))
    
    response = agent.json("...", schema=MySchema)
    
    assert response.output is not None
    assert response.output.name == "Test"
```

## Advanced Usage

### Raw JSON Schema

Use a dict instead of Pydantic:

```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "count": {"type": "integer"}
    },
    "required": ["name", "count"]
}

response = agent.json("...", schema=schema)
# response.output is a dict, not a Pydantic model
```

### Custom Validation

```python
from pydantic import validator

class Report(BaseModel):
    title: str
    sections: list[str]
    
    @validator("sections")
    def at_least_one_section(cls, v):
        if len(v) < 1:
            raise ValueError("Must have at least one section")
        return v
```

### Recursive Schemas

```python
from typing import Optional

class TreeNode(BaseModel):
    value: str
    children: list["TreeNode"] = []

TreeNode.model_rebuild()  # Required for recursive types
```

## Troubleshooting

### "Could not parse JSON"

The LLM returned text that isn't valid JSON. Try:
1. Use a more capable model
2. Add clearer instructions in system prompt
3. Simplify the schema

### "Schema validation failed"

The JSON is valid but doesn't match the schema:
1. Check field names match exactly
2. Verify types (string vs number)
3. Check required fields are present

### Inconsistent Results

LLM output can vary:
1. Lower temperature for more consistent output
2. Add examples in the prompt
3. Use native schema support (OpenAI, Gemini)

```python
response = agent.json(
    "...",
    schema=MySchema,
    temperature=0.1,  # More deterministic
)
```
