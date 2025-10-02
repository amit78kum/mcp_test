"""
Quick Start Example for PySR MCP Server
This is a minimal example to get you started quickly.
"""

import asyncio
import numpy as np
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def quick_start():
    """
    Quick demonstration: Discover y = x^2 + 1
    """
    
    print("🚀 PySR MCP Server - Quick Start Example")
    print("=" * 60)
    print("Goal: Discover the equation y = x² + 1 from data\n")
    
    # Connect to server
    server_params = StdioServerParameters(
        command="python",
        args=["pysr_mcp_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ Connected to PySR server\n")
            
            # Step 1: Generate synthetic data
            print("📊 Step 1: Generating data from y = x² + 1")
            X = np.linspace(-3, 3, 50).reshape(-1, 1)
            y = X.ravel()**2 + 1
            print(f"   Generated {len(X)} data points\n")
            
            # Step 2: Validate data
            print("🔍 Step 2: Validating data")
            result = await session.call_tool("validate_data", {
                "X_data": X.tolist(),
                "y_data": y.tolist()
            })
            print(result.content[0].text)
            
            # Step 3: Create model
            print("\n🏗️  Step 3: Creating PySR model")
            result = await session.call_tool("create_model", {
                "model_id": "quickstart",
                "niterations": 20,
                "binary_operators": ["+", "-", "*"],
                "unary_operators": ["square"],
                "maxsize": 10
            })
            print(result.content[0].text)
            
            # Step 4: Train model
            print("\n🎯 Step 4: Training model (please wait...)")
            result = await session.call_tool("fit_model", {
                "model_id": "quickstart",
                "X_data": X.tolist(),
                "y_data": y.tolist(),
                "variable_names": ["x"]
            })
            print(result.content[0].text)
            
            # Step 5: View discovered equations
            print("\n📝 Step 5: Top discovered equations")
            result = await session.call_tool("get_equations", {
                "model_id": "quickstart",
                "n_equations": 3
            })
            print(result.content[0].text)
            
            # Step 6: Make predictions
            print("\n🔮 Step 6: Testing predictions")
            test_points = [[0.0], [1.0], [2.0]]
            result = await session.call_tool("predict", {
                "model_id": "quickstart",
                "X_new": test_points
            })
            print(result.content[0].text)
            
            # Expected values
            print("\n   Expected values for comparison:")
            for x_val in [0.0, 1.0, 2.0]:
                expected = x_val**2 + 1
                print(f"   x={x_val} → y={expected}")
            
            # Step 7: Evaluate performance
            print("\n📊 Step 7: Model evaluation")
            result = await session.call_tool("evaluate_model", {
                "model_id": "quickstart"
            })
            print(result.content[0].text)
            
            # Step 8: Export equation
            print("\n📤 Step 8: Export best equation")
            result = await session.call_tool("export_equation", {
                "model_id": "quickstart",
                "equation_index": 0,
                "format": "latex"
            })
            print(result.content[0].text)
            
            print("\n" + "=" * 60)
            print("✅ Quick start completed successfully!")
            print("=" * 60)
            print("\n💡 Next steps:")
            print("   1. Try pysr_client.py for a comprehensive demo")
            print("   2. Experiment with your own datasets")
            print("   3. Adjust operators and parameters")
            print("   4. Read the README.md for advanced features")


if __name__ == "__main__":
    try:
        asyncio.run(quick_start())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  • Make sure Julia is installed: julia --version")
        print("  • Install PySR: python -c 'import pysr; pysr.install()'")
        print("  • Check requirements: pip install -r requirements.txt")