"""
MCP Client for PySR Server (Optimized for Speed)
Demonstrates how to connect and use the symbolic regression tools
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run_pysr_client():
    """Main client function to interact with PySR MCP server"""
    
    # Configure server parameters
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
        env=None
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            
            # Initialize the connection
            await session.initialize()
            
            print("=" * 60)
            print("Connected to PySR MCP Server")
            print("=" * 60)
            
            # List available tools
            tools = await session.list_tools()
            print(f"\n✓ Available Tools: {len(tools.tools)}")
            for tool in tools.tools:
                print(f"  • {tool.name}")
            
            print("\n" + "=" * 60)
            print("Example 1: Generate Sample Data")
            print("=" * 60)
            
            # Generate sample data
            result = await session.call_tool(
                "generate_sample_data",
                arguments={
                    "function": "quadratic",
                    "n_samples": 30,  # Smaller dataset for speed
                    "x_range": "[-5, 5]",
                    "noise_level": 0.3
                }
            )
            
            data_response = json.loads(result.content[0].text)
            print(f"✓ Generated {len(data_response['y'])} samples")
            print(f"  Function: {data_response['function']}")
            
            # Store X and y for next step
            X_data = json.dumps(data_response['X'])
            y_data = json.dumps(data_response['y'])
            
            print("\n" + "=" * 60)
            print("Example 2: Fit Symbolic Regression (Fast Settings)")
            print("=" * 60)
            print("⏳ Running regression (should take ~30 seconds)...")
            
            # Fit symbolic regression with FAST settings
            try:
                result = await asyncio.wait_for(
                    session.call_tool(
                        "fit_symbolic_regression",
                        arguments={
                            "x_data": X_data,
                            "y_data": y_data,
                            "niterations": 5,  # Very few iterations for speed
                            "binary_operators": "+,-,*",  # Simple operators only
                            "unary_operators": "",  # No unary operators for speed
                            "population_size": 15,
                            "max_complexity": 8
                        }
                    ),
                    timeout=90  # 90 second timeout
                )
                
                regression_response = json.loads(result.content[0].text)
                
                if regression_response["success"]:
                    print(f"\n✓ Best Equation Found:")
                    print(f"  {regression_response['best_equation']}")
                    
                    print(f"\n✓ Top Equations:")
                    for i, eq in enumerate(regression_response['equations'], 1):
                        print(f"  {i}. Loss: {eq['loss']:.6f} | {eq['equation']}")
                else:
                    print(f"✗ Error: {regression_response['error']}")
                    
            except asyncio.TimeoutError:
                print("✗ Timeout: Regression took too long. Try with smaller dataset.")
                return
            
            print("\n" + "=" * 60)
            print("Example 3: Predict with Known Equation")
            print("=" * 60)
            
            # Make predictions with the expected equation
            test_x = json.dumps([[-2.0], [0.0], [2.0], [4.0]])
            
            result = await session.call_tool(
                "predict_with_equation",
                arguments={
                    "equation": "x0**2 + 2*x0 + 1",
                    "x_data": test_x,
                    "feature_names": "x0"
                }
            )
            
            prediction_response = json.loads(result.content[0].text)
            
            if prediction_response["success"]:
                print("✓ Predictions for: x0² + 2x0 + 1")
                test_values = json.loads(test_x)
                for x_val, pred in zip(test_values, prediction_response['predictions']):
                    expected = x_val[0]**2 + 2*x_val[0] + 1
                    print(f"  x={x_val[0]:6.1f} → y={pred:8.2f} (expected: {expected:.2f})")
            else:
                print(f"✗ Error: {prediction_response['error']}")
            
            print("\n" + "=" * 60)
            print("Example 4: Simple Linear Regression")
            print("=" * 60)
            
            # Generate simple linear data
            result = await session.call_tool(
                "generate_sample_data",
                arguments={
                    "function": "linear",
                    "n_samples": 20,
                    "x_range": "[0, 10]",
                    "noise_level": 0.2
                }
            )
            
            linear_data = json.loads(result.content[0].text)
            print(f"✓ Generated linear data (y = 2x + 1)")
            
            # Quick fit
            print("⏳ Fitting linear equation...")
            try:
                result = await asyncio.wait_for(
                    session.call_tool(
                        "fit_symbolic_regression",
                        arguments={
                            "x_data": json.dumps(linear_data['X']),
                            "y_data": json.dumps(linear_data['y']),
                            "niterations": 3,
                            "binary_operators": "+,-,*",
                            "population_size": 10,
                            "max_complexity": 5
                        }
                    ),
                    timeout=60
                )
                
                linear_result = json.loads(result.content[0].text)
                if linear_result["success"]:
                    print(f"✓ Discovered: {linear_result['best_equation']}")
                    print(f"  (Expected: 2*x0 + 1)")
            except asyncio.TimeoutError:
                print("✗ Timeout on linear fit")
            
            print("\n" + "=" * 60)
            print("✓ Client Demo Complete!")
            print("=" * 60)
            print("\nTips for faster results:")
            print("  • Use fewer iterations (3-5)")
            print("  • Use simple operators: +,-,*")
            print("  • Avoid unary operators: sin, cos, exp")
            print("  • Use smaller datasets (20-50 samples)")


def main():
    """Run the async client"""
    try:
        asyncio.run(run_pysr_client())
    except KeyboardInterrupt:
        print("\n\n✗ Client stopped by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()