import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    """Main client function to interact with FastMCP server"""
    
    # Create server parameters for stdio connection
    server_params = StdioServerParameters(
        command="python",
        args=["fastmcp_server.py"],  # Path to your FastMCP server file
        env=None
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()
            print("✅ Connected to FastMCP server\n")
            
            # List available tools
            print("=" * 50)
            print("AVAILABLE TOOLS")
            print("=" * 50)
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"\n📦 {tool.name}")
                print(f"   {tool.description}")
            
            # List available resources
            print("\n" + "=" * 50)
            print("AVAILABLE RESOURCES")
            print("=" * 50)
            resources = await session.list_resources()
            for resource in resources.resources:
                print(f"\n📄 {resource.uri}")
                print(f"   {resource.name}")
            
            # List available prompts
            print("\n" + "=" * 50)
            print("AVAILABLE PROMPTS")
            print("=" * 50)
            prompts = await session.list_prompts()
            for prompt in prompts.prompts:
                print(f"\n📝 {prompt.name}")
                print(f"   {prompt.description}")
            
            # Example 1: Calculate
            print("\n" + "=" * 50)
            print("EXAMPLE 1: Calculator Tool")
            print("=" * 50)
            result = await session.call_tool("calculate", {
                "operation": "multiply",
                "a": 15,
                "b": 7
            })
            print(result.content[0].text)
            
            # Example 2: Random number
            print("\n" + "=" * 50)
            print("EXAMPLE 2: Random Number Generator")
            print("=" * 50)
            result = await session.call_tool("get_random_number", {
                "min_val": 1,
                "max_val": 50
            })
            print(result.content[0].text)
            
            # Example 3: Reverse string
            print("\n" + "=" * 50)
            print("EXAMPLE 3: String Reverser")
            print("=" * 50)
            result = await session.call_tool("reverse_string", {
                "text": "Hello FastMCP!"
            })
            print(result.content[0].text)
            
            # Example 4: Prime checker
            print("\n" + "=" * 50)
            print("EXAMPLE 4: Prime Number Checker")
            print("=" * 50)
            result = await session.call_tool("is_prime", {
                "number": 17
            })
            print(result.content[0].text)
            
            # Example 5: Word counter
            print("\n" + "=" * 50)
            print("EXAMPLE 5: Word Counter")
            print("=" * 50)
            result = await session.call_tool("count_words", {
                "text": "FastMCP makes building MCP servers incredibly easy! Try it today."
            })
            print(result.content[0].text)
            
            # Example 6: Get current time
            print("\n" + "=" * 50)
            print("EXAMPLE 6: Current Time")
            print("=" * 50)
            result = await session.call_tool("get_current_time", {
                "timezone": "UTC"
            })
            print(result.content[0].text)
            
            # Example 7: Read a resource
            print("\n" + "=" * 50)
            print("EXAMPLE 7: Read Resource")
            print("=" * 50)
            resource_content = await session.read_resource("config://settings")
            print(resource_content.contents[0].text)
            
            # Example 8: Use a prompt
            print("\n" + "=" * 50)
            print("EXAMPLE 8: Get Prompt Template")
            print("=" * 50)
            prompt_result = await session.get_prompt("code_review_prompt", {
                "language": "python",
                "code": "def add(a, b):\n    return a + b"
            })
            print(prompt_result.messages[0].content.text)
            
            print("\n" + "=" * 50)
            print("✅ All examples completed!")
            print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())