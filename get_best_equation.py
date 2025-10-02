"""
Minimal example showing how to access PySR MCP server endpoint
to get the best equation from data.
"""

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def get_best_equation_from_data(X, y, variable_names=None):
    """
    Simple function to get best equation from data using MCP server
    
    Args:
        X: Input data (list of lists)
        y: Target data (list)
        variable_names: Optional variable names
    
    Returns:
        Best equation as string
    """
    
    # Connect to MCP server
    server_params = StdioServerParameters(
        command="python",
        args=["simple_pysr_mcp_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            
            # Initialize connection
            await session.initialize()
            
            # Call the tool to train and get best equation
            result = await session.call_tool(
                "train_and_get_best_equation",
                {
                    "X_data": X,
                    "y_data": y,
                    "model_id": "my_model",
                    "niterations": 10,
                    "variable_names": variable_names
                }
            )
            
            # Return the result
            return result.content[0].text


async def main():
    """
    Example usage: Discover equation from y = x² + 2x + 1
    """
    
    print("=" * 60)
    print("Minimal Example: Get Best Equation from Data")
    print("=" * 60)
    
    # Your data
    X = [[-3], [-2], [-1], [0], [1], [2], [3], [4], [5]]
    y = [4, 1, 0, 1, 4, 9, 16, 25, 36]  # This is y = x² + 2x + 1
    
    print("\nData: y = x² + 2x + 1")
    print(f"X: {X}")
    print(f"y: {y}")
    
    print("\n🔬 Discovering equation from data...")
    print("(First run may take 30-60 seconds)\n")
    
    # Get best equation
    equation = await get_best_equation_from_data(
        X=X,
        y=y,
        variable_names=["x"]
    )
    
    print(equation)
    
    print("\n" + "=" * 60)
    print("✅ Done!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())