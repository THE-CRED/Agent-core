"""
Tool usage examples.

Demonstrates how to create and use tools with Agent.
"""

import json

from pydantic import BaseModel

from agent import Agent, tool

# ============================================================================
# Basic Tool
# ============================================================================

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Simulated weather data
    weather_data = {
        "new york": {"temp": 72, "condition": "sunny"},
        "london": {"temp": 55, "condition": "cloudy"},
        "tokyo": {"temp": 68, "condition": "partly cloudy"},
    }

    data = weather_data.get(city.lower(), {"temp": 70, "condition": "unknown"})
    return f"Weather in {city}: {data['temp']}F, {data['condition']}"


def basic_tool_usage():
    """Basic tool calling example."""
    agent = Agent(
        provider="openai",
        model="gpt-4o",
        tools=[get_weather],
    )

    response = agent.run("What's the weather like in Tokyo?")
    print(response.text)


# ============================================================================
# Calculator Tool
# ============================================================================

@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression.

    Args:
        expression: A math expression like "2 + 2" or "sqrt(16)"
    """
    import math

    # Safe evaluation with limited scope
    allowed_names = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "round": round,
    }

    try:
        # Only allow safe operations
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def calculator_example():
    """Math calculation with tools."""
    agent = Agent(
        provider="anthropic",
        model="claude-sonnet",
        tools=[calculate],
    )

    questions = [
        "What is 15% of 250?",
        "Calculate the square root of 144",
        "What is sin(pi/2)?",
    ]

    for q in questions:
        response = agent.run(q)
        print(f"Q: {q}")
        print(f"A: {response.text}")
        print()


# ============================================================================
# Multiple Tools
# ============================================================================

@tool
def search_database(query: str) -> str:
    """Search the product database."""
    # Simulated database
    products = [
        {"id": 1, "name": "Laptop Pro", "price": 1299, "stock": 50},
        {"id": 2, "name": "Wireless Mouse", "price": 49, "stock": 200},
        {"id": 3, "name": "USB-C Hub", "price": 79, "stock": 75},
    ]

    results = [p for p in products if query.lower() in p["name"].lower()]
    if results:
        return json.dumps(results, indent=2)
    return "No products found"


@tool
def check_inventory(product_id: int) -> str:
    """Check inventory for a specific product."""
    inventory = {1: 50, 2: 200, 3: 75}
    stock = inventory.get(product_id, 0)
    return f"Product {product_id} has {stock} units in stock"


@tool
def calculate_discount(price: float, discount_percent: float) -> str:
    """Calculate discounted price."""
    discounted = price * (1 - discount_percent / 100)
    return f"Original: ${price:.2f}, Discount: {discount_percent}%, Final: ${discounted:.2f}"


def multi_tool_example():
    """Using multiple tools together."""
    agent = Agent(
        provider="openai",
        model="gpt-4o",
        tools=[search_database, check_inventory, calculate_discount],
    )

    response = agent.run(
        "Find laptops in the database, check their inventory, "
        "and calculate a 15% discount on the price"
    )
    print(response.text)


# ============================================================================
# Async Tool
# ============================================================================

@tool
async def fetch_url(url: str) -> str:
    """Fetch content from a URL (simulated)."""
    # In real usage, you'd use aiohttp or similar
    return f"Content from {url}: [simulated content]"


async def async_tool_example():
    """Using async tools."""
    agent = Agent(
        provider="openai",
        model="gpt-4o",
        tools=[fetch_url],
    )

    response = await agent.run_async("Fetch the content from example.com")
    print(response.text)


# ============================================================================
# Tool with Complex Types
# ============================================================================

class SearchFilters(BaseModel):
    category: str
    min_price: float
    max_price: float


@tool
def advanced_search(query: str, min_price: float = 0, max_price: float = 10000) -> str:
    """
    Search products with filters.

    Args:
        query: Search query
        min_price: Minimum price filter
        max_price: Maximum price filter
    """
    # Simulated search
    return f"Found 5 products matching '{query}' between ${min_price} and ${max_price}"


def complex_tool_example():
    """Tools with multiple parameters."""
    agent = Agent(
        provider="anthropic",
        model="claude-sonnet",
        tools=[advanced_search],
    )

    response = agent.run("Find laptops under $1000")
    print(response.text)


if __name__ == "__main__":
    # Uncomment to run:
    # basic_tool_usage()
    # calculator_example()
    # multi_tool_example()
    # complex_tool_example()

    # For async:
    # import asyncio
    # asyncio.run(async_tool_example())

    print("Uncomment an example function to run it!")
