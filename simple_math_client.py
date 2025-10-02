"""
Fast MCP Client - Works in seconds instead of minutes!
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run_fast_demo():
    """Fast demo that completes in seconds"""
    
    # Configure server parameters
    server_params = StdioServerParameters(
        command="python",
        args=["simple_math_server.py"],  # Use the new server
        env=None
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            
            # Initialize the connection
            await session.initialize()
            
            print("=" * 60)
            print("🚀 Connected to Simple Math MCP Server")
            print("=" * 60)
            
            # List available tools
            tools = await session.list_tools()
            print(f"\n✓ Available Tools: {len(tools.tools)}")
            for tool in tools.tools:
                print(f"  • {tool.name}")
            
            # Example 1: Generate Data
            print("\n" + "=" * 60)
            print("📊 Example 1: Generate Quadratic Data")
            print("=" * 60)
            
            result = await session.call_tool(
                "generate_sample_data",
                arguments={
                    "function": "quadratic",
                    "n_samples": 30,
                    "x_range": "[-5, 5]",
                    "noise_level": 0.5
                }
            )
            
            data = json.loads(result.content[0].text)
            print(f"✓ Generated {data['n_samples']} samples")
            print(f"  True equation: {data['true_equation']}")
            print(f"  Noise level: {data['noise_level']}")
            
            # Example 2: Polynomial Fit (INSTANT!)
            print("\n" + "=" * 60)
            print("🔍 Example 2: Polynomial Fit (Instant!)")
            print("=" * 60)
            
            result = await session.call_tool(
                "polynomial_fit",
                arguments={
                    "x_data": json.dumps(data['X']),
                    "y_data": json.dumps(data['y']),
                    "degree": 2
                }
            )
            
            fit = json.loads(result.content[0].text)
            if fit['success']:
                print(f"✓ Discovered Equation:")
                print(f"  {fit['equation']}")
                print(f"  R² = {fit['r_squared']:.4f}")
                print(f"  MSE = {fit['mean_squared_error']:.4f}")
            
            # Example 3: Evaluate Expression
            print("\n" + "=" * 60)
            print("🧮 Example 3: Evaluate Custom Expression")
            print("=" * 60)
            
            test_x = json.dumps([-2, 0, 2, 4])
            
            result = await session.call_tool(
                "evaluate_expression",
                arguments={
                    "expression": "x**2 + 2*x + 1",
                    "x_data": test_x
                }
            )
            
            eval_result = json.loads(result.content[0].text)
            if eval_result['success']:
                print(f"✓ Expression: {eval_result['expression']}")
                for x, y in zip(eval_result['x_values'], eval_result['y_values']):
                    print(f"  x={x:6.1f} → y={y:8.2f}")
            
            # Example 4: Statistics
            print("\n" + "=" * 60)
            print("📈 Example 4: Calculate Statistics")
            print("=" * 60)
            
            result = await session.call_tool(
                "calculate_statistics",
                arguments={
                    "data": json.dumps(data['y'])
                }
            )
            
            stats = json.loads(result.content[0].text)
            if stats['success']:
                print(f"✓ Dataset Statistics:")
                print(f"  Count:  {stats['count']}")
                print(f"  Mean:   {stats['mean']:.3f}")
                print(f"  Median: {stats['median']:.3f}")
                print(f"  Std:    {stats['std']:.3f}")
                print(f"  Range:  [{stats['min']:.3f}, {stats['max']:.3f}]")
            
            # Example 5: Test Different Functions
            print("\n" + "=" * 60)
            print("🎯 Example 5: Fit Multiple Functions")
            print("=" * 60)
            
            functions = ['linear', 'cubic']
            
            for func in functions:
                # Generate data
                result = await session.call_tool(
                    "generate_sample_data",
                    arguments={
                        "function": func,
                        "n_samples": 20,
                        "x_range": "[-3, 3]",
                        "noise_level": 0.2
                    }
                )
                func_data = json.loads(result.content[0].text)
                
                # Fit polynomial
                degree = 1 if func == 'linear' else 3
                result = await session.call_tool(
                    "polynomial_fit",
                    arguments={
                        "x_data": json.dumps(func_data['X']),
                        "y_data": json.dumps(func_data['y']),
                        "degree": degree
                    }
                )
                func_fit = json.loads(result.content[0].text)
                
                print(f"\n{func.title()} Function:")
                print(f"  True:       {func_data['true_equation']}")
                if func_fit['success']:
                    print(f"  Discovered: {func_fit['equation']}")
                    print(f"  R²:         {func_fit['r_squared']:.4f}")
            
            # Example 6: Expression with Functions
            print("\n" + "=" * 60)
            print("🌊 Example 6: Evaluate Sin Wave")
            print("=" * 60)
            
            result = await session.call_tool(
                "evaluate_expression",
                arguments={
                    "expression": "sin(x) + 0.5*cos(2*x)",
                    "x_data": json.dumps([0, 0.785, 1.57, 3.14, 6.28])
                }
            )
            
            wave = json.loads(result.content[0].text)
            if wave['success']:
                print(f"✓ Expression: {wave['expression']}")
                for x, y in zip(wave['x_values'], wave['y_values']):
                    print(f"  x={x:.2f} → y={y:7.3f}")
            
            print("\n" + "=" * 60)
            print("✅ All Examples Completed Successfully!")
            print("=" * 60)
            print(f"\n⚡ Total Time: Just a few seconds!")
            print("\n💡 This demonstrates MCP working perfectly.")
            print("   PySR symbolic regression is optional and slower.")


def main():
    """Run the async client"""
    try:
        asyncio.run(run_fast_demo())
    except KeyboardInterrupt:
        print("\n\n✗ Stopped by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()