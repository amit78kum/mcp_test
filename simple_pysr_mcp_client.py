import asyncio
import numpy as np
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def demo_best_equation():
    """
    Demo: Connect to PySR MCP server and get best equation
    """
    
    print("=" * 70)
    print("🔬 PySR MCP Client - Best Equation Demo")
    print("=" * 70)
    
    # Configure server connection
    server_params = StdioServerParameters(
        command="python",
        args=["simple_pysr_mcp_server.py"],
        env=None
    )
    
    # Connect to server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            
            # Initialize session
            await session.initialize()
            print("\n✅ Connected to PySR MCP Server\n")
            
            # List available tools
            print("📦 Available Tools:")
            print("-" * 70)
            tools = await session.list_tools()
            for i, tool in enumerate(tools.tools, 1):
                print(f"{i}. {tool.name}")
                print(f"   └─ {tool.description}\n")
            
            # ═══════════════════════════════════════════════════════════════
            # DEMO 1: Simple quadratic equation (y = 2x² + 3x + 1)
            # ═══════════════════════════════════════════════════════════════
            print("\n" + "=" * 70)
            print("📊 DEMO 1: Discover Quadratic Equation")
            print("=" * 70)
            print("Goal: Find equation from y = 2x² + 3x + 1\n")
            
            # Generate data
            np.random.seed(42)
            X_quad = np.linspace(-5, 5, 50).reshape(-1, 1)
            y_quad = 2 * X_quad.ravel()**2 + 3 * X_quad.ravel() + 1
            
            print("📈 Training model on quadratic data...")
            print("   (This may take 30-60 seconds)\n")
            
            # Train and get best equation
            result = await session.call_tool(
                "train_and_get_best_equation",
                {
                    "X_data": X_quad.tolist(),
                    "y_data": y_quad.tolist(),
                    "model_id": "quadratic_model",
                    "niterations": 10,
                    "variable_names": ["x"]
                }
            )
            
            print(result.content[0].text)
            
            # ═══════════════════════════════════════════════════════════════
            # DEMO 2: Get best equation again (without retraining)
            # ═══════════════════════════════════════════════════════════════
            print("\n" + "=" * 70)
            print("📊 DEMO 2: Retrieve Best Equation (No Retraining)")
            print("=" * 70)
            
            result = await session.call_tool(
                "get_best_equation",
                {"model_id": "quadratic_model"}
            )
            
            print(result.content[0].text)
            
            # ═══════════════════════════════════════════════════════════════
            # DEMO 3: Get top 5 equations
            # ═══════════════════════════════════════════════════════════════
            print("\n" + "=" * 70)
            print("📊 DEMO 3: Get Top 5 Equations")
            print("=" * 70)
            
            result = await session.call_tool(
                "get_top_equations",
                {
                    "model_id": "quadratic_model",
                    "n": 5
                }
            )
            
            print(result.content[0].text)
            
            # ═══════════════════════════════════════════════════════════════
            # DEMO 4: Make predictions with best equation
            # ═══════════════════════════════════════════════════════════════
            print("\n" + "=" * 70)
            print("📊 DEMO 4: Make Predictions")
            print("=" * 70)
            
            test_points = [[-2.0], [0.0], [1.0], [3.0], [5.0]]
            
            result = await session.call_tool(
                "predict_with_best_equation",
                {
                    "model_id": "quadratic_model",
                    "X_new": test_points
                }
            )
            
            print(result.content[0].text)
            
            # Show expected values
            print("Expected values for y = 2x² + 3x + 1:")
            for point in test_points:
                x = point[0]
                expected = 2 * x**2 + 3 * x + 1
                print(f"  x={x:5.1f} → y={expected:8.2f}")
            
            # ═══════════════════════════════════════════════════════════════
            # DEMO 5: Export equation in different formats
            # ═══════════════════════════════════════════════════════════════
            print("\n" + "=" * 70)
            print("📊 DEMO 5: Export Equation")
            print("=" * 70)
            
            for fmt in ["latex", "sympy", "python"]:
                print(f"\n🔹 Format: {fmt.upper()}")
                print("-" * 70)
                result = await session.call_tool(
                    "export_best_equation",
                    {
                        "model_id": "quadratic_model",
                        "format": fmt
                    }
                )
                print(result.content[0].text)
            
            # ═══════════════════════════════════════════════════════════════
            # DEMO 6: Multi-variable equation (z = x² + y²)
            # ═══════════════════════════════════════════════════════════════
            print("\n" + "=" * 70)
            print("📊 DEMO 6: Multi-Variable Equation")
            print("=" * 70)
            print("Goal: Find equation from z = x² + y²\n")
            
            # Generate 2D data
            np.random.seed(123)
            X_multi = np.random.uniform(-3, 3, (40, 2))
            y_multi = X_multi[:, 0]**2 + X_multi[:, 1]**2
            
            print("📈 Training model on 2-variable data...")
            print("   (This may take 30-60 seconds)\n")
            
            result = await session.call_tool(
                "train_and_get_best_equation",
                {
                    "X_data": X_multi.tolist(),
                    "y_data": y_multi.tolist(),
                    "model_id": "multivariable_model",
                    "niterations": 10,
                    "variable_names": ["x", "y"]
                }
            )
            
            print(result.content[0].text)
            
            # Test predictions
            print("\n🔮 Testing predictions:")
            test_multi = [[1.0, 1.0], [2.0, 2.0], [3.0, 0.0]]
            
            result = await session.call_tool(
                "predict_with_best_equation",
                {
                    "model_id": "multivariable_model",
                    "X_new": test_multi
                }
            )
            
            print(result.content[0].text)
            
            print("Expected values for z = x² + y²:")
            for point in test_multi:
                x, y = point
                expected = x**2 + y**2
                print(f"  x={x:.1f}, y={y:.1f} → z={expected:.2f}")
            
            # ═══════════════════════════════════════════════════════════════
            # DEMO 7: List all models
            # ═══════════════════════════════════════════════════════════════
            print("\n" + "=" * 70)
            print("📊 DEMO 7: List All Models")
            print("=" * 70)
            
            resource = await session.read_resource("models://list")
            print(resource.contents[0].text)
            
            # ═══════════════════════════════════════════════════════════════
            # Summary
            # ═══════════════════════════════════════════════════════════════
            print("\n" + "=" * 70)
            print("✅ All Demonstrations Completed!")
            print("=" * 70)
            
            print("\n💡 What We Demonstrated:")
            print("  ✓ Train model and get best equation automatically")
            print("  ✓ Retrieve best equation from trained model")
            print("  ✓ Get top N equations ranked by score")
            print("  ✓ Make predictions using discovered equations")
            print("  ✓ Export equations in LaTeX, SymPy, and Python")
            print("  ✓ Handle multi-variable equations")
            print("  ✓ List all trained models")
            
            print("\n🎯 Key Features:")
            print("  • Automatic equation discovery from data")
            print("  • Multiple equation candidates ranked by complexity/accuracy")
            print("  • Easy-to-use MCP interface")
            print("  • Export to various formats for different use cases")
            print("  • Persistent models during server session")
            
            print("\n" + "=" * 70)


async def quick_example():
    """
    Quick example: Just train and get the best equation
    """
    
    print("🚀 Quick Example: Train and Get Best Equation\n")
    
    server_params = StdioServerParameters(
        command="python",
        args=["simple_pysr_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Simple data: y = x² + 1
            X = [[x] for x in range(-5, 6)]
            y = [x[0]**2 + 1 for x in X]
            
            print("Training on: y = x² + 1\n")
            
            result = await session.call_tool(
                "train_and_get_best_equation",
                {
                    "X_data": X,
                    "y_data": y,
                    "model_id": "quick_model",
                    "niterations": 5,
                    "variable_names": ["x"]
                }
            )
            
            print(result.content[0].text)


if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 70)
    print("🔬 PySR MCP Client")
    print("=" * 70)
    print("\nChoose demo mode:")
    print("  1. Quick example (fast, ~1 minute)")
    print("  2. Full demonstration (comprehensive, ~3-5 minutes)")
    print()
    
    choice = input("Enter choice (1 or 2, default=1): ").strip() or "1"
    
    try:
        if choice == "1":
            asyncio.run(quick_example())
        else:
            asyncio.run(demo_best_equation())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("  1. Make sure server file is named: simple_pysr_server.py")
        print("  2. Ensure Julia and PySR are properly installed")
        print("  3. Run: python pysr_debug.py")
        import traceback
        traceback.print_exc()