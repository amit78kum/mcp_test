from fastmcp import FastMCP
import random
import math
from datetime import datetime

# Create FastMCP server instance
mcp = FastMCP("Example FastMCP Server")

# Tool 1: Calculator
@mcp.tool()
def calculate(operation: str, a: float, b: float) -> str:
    """
    Perform basic arithmetic operations.
    
    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        a: First number
        b: Second number
    
    Returns:
        The result of the calculation
    """
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else "Error: Division by zero"
    }
    
    if operation not in operations:
        return f"Error: Unknown operation '{operation}'. Use: add, subtract, multiply, divide"
    
    result = operations[operation](a, b)
    return f"{a} {operation} {b} = {result}"


# Tool 2: Random number generator
@mcp.tool()
def get_random_number(min_val: int = 0, max_val: int = 100) -> str:
    """
    Generate a random number within a specified range.
    
    Args:
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)
    
    Returns:
        A random number in the specified range
    """
    num = random.randint(min_val, max_val)
    return f"Random number between {min_val} and {max_val}: {num}"


# Tool 3: String reverser
@mcp.tool()
def reverse_string(text: str) -> str:
    """
    Reverse a given string.
    
    Args:
        text: The string to reverse
    
    Returns:
        The reversed string
    """
    reversed_text = text[::-1]
    return f'Original: "{text}"\nReversed: "{reversed_text}"'


# Tool 4: Prime number checker
@mcp.tool()
def is_prime(number: int) -> str:
    """
    Check if a number is prime.
    
    Args:
        number: The number to check
    
    Returns:
        Whether the number is prime or not
    """
    if number < 2:
        return f"{number} is not prime"
    
    for i in range(2, int(math.sqrt(number)) + 1):
        if number % i == 0:
            return f"{number} is not prime (divisible by {i})"
    
    return f"{number} is prime!"


# Tool 5: Get current time
@mcp.tool()
def get_current_time(timezone: str = "UTC") -> str:
    """
    Get the current date and time.
    
    Args:
        timezone: Timezone (currently only supports UTC)
    
    Returns:
        Current date and time
    """
    now = datetime.utcnow()
    return f"Current time ({timezone}): {now.strftime('%Y-%m-%d %H:%M:%S')}"


# Tool 6: Word counter
@mcp.tool()
def count_words(text: str) -> str:
    """
    Count words, characters, and sentences in text.
    
    Args:
        text: The text to analyze
    
    Returns:
        Statistics about the text
    """
    words = len(text.split())
    chars = len(text)
    chars_no_spaces = len(text.replace(" ", ""))
    sentences = text.count('.') + text.count('!') + text.count('?')
    
    return f"""Text Analysis:
- Words: {words}
- Characters (with spaces): {chars}
- Characters (without spaces): {chars_no_spaces}
- Sentences: {sentences}"""


# Resource: Static data example
@mcp.resource("config://settings")
def get_settings() -> str:
    """Server configuration settings"""
    return """Server Settings:
- Version: 1.0.0
- Max connections: 100
- Timeout: 30s
- Debug mode: Enabled"""


# Resource: Dynamic data example
@mcp.resource("data://stats")
def get_stats() -> str:
    """Server statistics"""
    return f"""Server Statistics:
- Uptime: Active
- Requests processed: {random.randint(100, 1000)}
- Last restart: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"""


# Prompt: Example template
@mcp.prompt()
def code_review_prompt(language: str = "python", code: str = "") -> str:
    """
    Generate a code review prompt.
    
    Args:
        language: Programming language
        code: Code to review
    """
    return f"""Please review this {language} code and provide feedback:

```{language}
{code}
```

Focus on:
1. Code quality and best practices
2. Potential bugs or issues
3. Performance considerations
4. Suggestions for improvement"""


if __name__ == "__main__":
    # Run the server
    mcp.run()