# Structured Outputs

Agent supports type-safe structured outputs using Pydantic models. This ensures LLM responses conform to your expected schema.

## Basic Usage

```python
from pydantic import BaseModel
from agent import Agent

class Sentiment(BaseModel):
    text: str
    sentiment: str  # "positive", "negative", "neutral"
    confidence: float

agent = Agent(provider="openai", model="gpt-4o")
response = agent.json(
    "Analyze the sentiment: 'I love this product!'",
    schema=Sentiment
)

# response.output is a Sentiment instance
print(response.output.sentiment)    # "positive"
print(response.output.confidence)   # 0.95
```

## Complex Schemas

### Nested Models

```python
from pydantic import BaseModel

class Address(BaseModel):
    street: str
    city: str
    country: str
    postal_code: str

class Person(BaseModel):
    name: str
    age: int
    email: str
    address: Address

response = agent.json(
    "Extract person info from: John Doe, 30, john@example.com, lives at 123 Main St, NYC, USA 10001",
    schema=Person
)

print(response.output.address.city)  # "NYC"
```

### Lists and Optional Fields

```python
from pydantic import BaseModel
from typing import Optional

class Task(BaseModel):
    title: str
    description: str
    priority: str  # "high", "medium", "low"
    due_date: Optional[str] = None
    tags: list[str] = []

class TaskList(BaseModel):
    tasks: list[Task]
    total_count: int

response = agent.json(
    "Create a task list for launching a new product",
    schema=TaskList
)

for task in response.output.tasks:
    print(f"- [{task.priority}] {task.title}")
```

### Enums and Literals

```python
from pydantic import BaseModel
from typing import Literal
from enum import Enum

class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class Issue(BaseModel):
    title: str
    priority: Priority
    status: Literal["open", "in_progress", "closed"]
    
response = agent.json("Create a high priority bug report for login issues", schema=Issue)
print(response.output.priority)  # Priority.HIGH
```

## Field Descriptions

Add descriptions to help the LLM understand your schema:

```python
from pydantic import BaseModel, Field

class CodeReview(BaseModel):
    """A code review assessment."""
    
    summary: str = Field(
        description="Brief summary of the code changes"
    )
    issues: list[str] = Field(
        description="List of potential issues or bugs found"
    )
    suggestions: list[str] = Field(
        description="Improvement suggestions for better code quality"
    )
    score: int = Field(
        ge=1, le=10,
        description="Overall code quality score from 1-10"
    )
    approved: bool = Field(
        description="Whether the code is ready for merge"
    )
```

## Validation

Pydantic validates responses automatically:

```python
from pydantic import BaseModel, Field, field_validator

class Rating(BaseModel):
    score: int = Field(ge=1, le=5)
    review: str = Field(min_length=10)
    
    @field_validator('review')
    @classmethod
    def review_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Review cannot be empty')
        return v.strip()

# If LLM returns invalid data, SchemaValidationError is raised
try:
    response = agent.json("Rate this product: Amazing!", schema=Rating)
except SchemaValidationError as e:
    print(f"Invalid response: {e}")
```

## Native vs Prompt-Based

### Native Schema Support (OpenAI, Gemini)

Some providers support schema-enforced generation:

```python
# OpenAI uses response_format with json_schema
agent = Agent(provider="openai", model="gpt-4o")
response = agent.json("...", schema=MyModel)
# Uses native JSON schema mode - guaranteed valid JSON
```

### Prompt-Based (Anthropic, others)

For providers without native support, Agent adds schema instructions to the prompt:

```python
# Anthropic uses prompt engineering
agent = Agent(provider="anthropic", model="claude-sonnet")
response = agent.json("...", schema=MyModel)
# Agent adds: "Respond with JSON matching this schema: {...}"
```

## JSON Schema Directly

You can also use raw JSON Schema:

```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "tags": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["name", "age"]
}

response = agent.json("Create a user profile", schema=schema)
# response.output is a dict
```

## With Sessions

Use structured outputs in multi-turn conversations:

```python
session = agent.session()

# Regular messages
session.run("I want to plan a trip to Japan")

# Structured output for final result
class TravelPlan(BaseModel):
    destination: str
    duration_days: int
    activities: list[str]
    estimated_budget: float

response = session.json(
    "Create a detailed travel plan based on our discussion",
    schema=TravelPlan
)
```

## Async Support

```python
response = await agent.json_async(
    "Analyze this data",
    schema=AnalysisResult
)
```

## Error Handling

```python
from agent.errors import SchemaValidationError

try:
    response = agent.json("...", schema=MyModel)
except SchemaValidationError as e:
    print(f"Schema: {e.schema}")
    print(f"Output: {e.output}")
    print(f"Message: {e.message}")
```

## Repair Attempts

Agent can attempt to fix malformed JSON:

```python
from agent.schemas import Schema

schema = Schema(
    MyModel,
    strict=True,
    repair_attempts=2  # Try to fix JSON twice before failing
)

# Use with the Schema wrapper for more control
from agent.execution.structured_output import StructuredOutputHandler
handler = StructuredOutputHandler(MyModel, repair_attempts=3)
```

## Best Practices

### 1. Keep Schemas Simple

```python
# Good - flat, clear structure
class Summary(BaseModel):
    title: str
    points: list[str]
    word_count: int

# Avoid - deeply nested, complex
class Analysis(BaseModel):
    meta: Meta
    sections: list[Section]  # where Section has SubSections...
```

### 2. Use Descriptive Field Names

```python
# Good
class Product(BaseModel):
    product_name: str
    price_usd: float
    in_stock: bool

# Bad
class Product(BaseModel):
    n: str
    p: float
    s: bool
```

### 3. Provide Examples in Descriptions

```python
class Category(BaseModel):
    name: str = Field(
        description="Category name, e.g., 'Electronics', 'Clothing', 'Books'"
    )
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0"
    )
```

### 4. Handle Optional Fields

```python
class Article(BaseModel):
    title: str
    content: str
    author: str | None = None  # May not always be available
    published_date: str | None = None
    tags: list[str] = []  # Empty list as default
```

## Next Steps

- [Tools](tools.md) - Combine structured outputs with tool calling
- [Sessions](sessions.md) - Structured outputs in conversations
- [Type System](types.md) - Understanding Agent's type system
