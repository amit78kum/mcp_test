"""
Basic PySR MCP Client
Connects to PySR server and runs symbolic regression examples
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run_basic_pysr():
    """Run basic PySR examples"""
    
    # Server configuration
    server_params = StdioServerParameters(
        command="python",
        args=["basic_pysr_server.py"],
        env=None
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            
            # Initialize
            await session.initialize()
            
            print("=" * 60)
            print("🔬 Connected to Basic PySR MCP Server")
            print("=" * 60)
            
            # List tools
            tools = await session.list_tools()
            print(f"\n✓ Available Tools: {len(tools.tools)}")
            for tool in tools.tools:
                print(f"  • {tool.name}")
            
            # Example 1: Generate test data
            print("\n" + "=" * 60)
            print("📊 Example 1: Generate Quadratic Data")
            print("=" * 60)
            
            result = await session.call_tool(
                "create_test_data",
                arguments={
                    "equation_type": "quadratic",
                    "n_points": 20
                }
            )
            
            data = json.loads(result.content[0].text)
            
            if data["success"]:
                print(f"✓ Generated {data['n_points']} data points")
                print(f"  True equation: {data['true_equation']}")
                x_data = json.dumps(data['x_data'])
                y_data = json.dumps(data['y_data'])
            else:
                print(f"✗ Error: {data['error']}")
                return
            
            # Example 2: Run PySR
            print("\n" + "=" * 60)
            print("🧬 Example 2: Symbolic Regression")
            print("=" * 60)
            print("⏳ Running PySR (this may take 30-90 seconds)...")
            print("   First run: ~2-3 minutes (Julia compilation)")
            print("   Later runs: ~30-60 seconds")
            print("")
            
            try:
                result = await asyncio.wait_for(
                    session.call_tool(
                        "symbolic_regression",
                        arguments={
                            "x_data": x_data,
                            "y_data": y_data,
                            "niterations": 3  # Keep very low for speed
                        }
                    ),
                    timeout=360 # 3 minute timeout
                )
                
                pysr_result = json.loads(result.content[0].text)
                
                if pysr_result["success"]:
                    print("\n✅ PySR Completed!")
                    print("=" * 60)
                    print(f"🎯 Best Equation: {pysr_result['best_equation']}")
                    print(f"   True Equation:  {data['true_equation']}")
                    print("")
                    print(f"📋 Found {pysr_result['n_equations']} equations")
                    print("\nTop 3 Equations:")
                    for i, eq in enumerate(pysr_result['top_equations'], 1):
                        print(f"  {i}. {eq['equation']}")
                        print(f"     Complexity: {eq['complexity']}, Loss: {eq['loss']:.6f}")
                else:
                    print(f"\n✗ PySR Error: {pysr_result['error']}")
                    
            except asyncio.TimeoutError:
                print("\n✗ Timeout: PySR took longer than 3 minutes")
                print("   This is normal on first run (Julia compilation)")
                print("   Try running again - it will be faster!")
            
            # Example 3: Try linear (faster)
            print("\n" + "=" * 60)
            print("📈 Example 3: Linear Regression (Faster)")
            print("=" * 60)
            
            result = await session.call_tool(
                "create_test_data",
                arguments={
                    "equation_type": "linear",
                    "n_points": 15
                }
            )
            
            linear_data = json.loads(result.content[0].text)
            
            if linear_data["success"]:
                print(f"✓ Generated linear data: {linear_data['true_equation']}")
                print("⏳ Running PySR...")
                
                try:
                    result = await asyncio.wait_for(
                        session.call_tool(
                            "symbolic_regression",
                            arguments={
                                "x_data": json.dumps(linear_data['x_data']),
                                "y_data": json.dumps(linear_data['y_data']),
                                "niterations": 3
                            }
                        ),
                        timeout=120
                    )
                    
                    linear_result = json.loads(result.content[0].text)
                    
                    if linear_result["success"]:
                        print(f"\n✓ Discovered: {linear_result['best_equation']}")
                        print(f"  Expected:   {linear_data['true_equation']}")
                        
                except asyncio.TimeoutError:
                    print("✗ Timeout on linear regression")
            
            print("\n" + "=" * 60)
            print("✅ Demo Complete!")
            print("=" * 60)
            print("\n💡 Tips for faster PySR:")
            print("  • Use niterations=3-5 (not 10+)")
            print("  • Keep n_points small (15-30)")
            print("  • Simple equations are found faster")
            print("  • First run is always slow (Julia compile)")


def main():
    """Run the client"""
    try:
        asyncio.run(run_basic_pysr())
    except KeyboardInterrupt:
        print("\n\n✗ Stopped by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()