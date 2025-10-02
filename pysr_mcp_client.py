import asyncio
import numpy as np
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    """Comprehensive demo of PySR MCP server capabilities"""
    
    # Create server parameters
    server_params = StdioServerParameters(
        command="python",
        args=["pysr_mcp_server.py"],
        env=None
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()
            print("=" * 70)
            print("🚀 Connected to PySR MCP Server")
            print("=" * 70)
            
            # List available tools
            print("\n📦 AVAILABLE TOOLS")
            print("-" * 70)
            tools = await session.list_tools()
            for i, tool in enumerate(tools.tools, 1):
                print(f"{i:2d}. {tool.name}")
                print(f"    {tool.description}")
            
            # List available resources
            print("\n📄 AVAILABLE RESOURCES")
            print("-" * 70)
            resources = await session.list_resources()
            for resource in resources.resources:
                print(f"• {resource.uri}: {resource.name}")
            
            print("\n" + "=" * 70)
            print("🔬 DEMONSTRATION: Complete PySR Workflow")
            print("=" * 70)
            
            # Step 1: Generate synthetic data (quadratic function: y = 2*x^2 + 3*x + 1)
            print("\n[1] Generating synthetic data: y = 2*x² + 3*x + 1")
            np.random.seed(42)
            X_train = np.random.uniform(-5, 5, 100).reshape(-1, 1)
            y_train = 2 * X_train.ravel()**2 + 3 * X_train.ravel() + 1 + np.random.normal(0, 0.5, 100)
            
            X_test = np.random.uniform(-5, 5, 20).reshape(-1, 1)
            y_test = 2 * X_test.ravel()**2 + 3 * X_test.ravel() + 1
            
            print(f"    Training data: {X_train.shape[0]} samples")
            print(f"    Test data: {X_test.shape[0]} samples")
            
            # Step 2: Validate data
            print("\n[2] Validating training data...")
            print("-" * 70)
            result = await session.call_tool("validate_data", {
                "X_data": X_train.tolist(),
                "y_data": y_train.tolist()
            })
            print(result.content[0].text)
            
            # Step 3: Create model
            print("\n[3] Creating PySR model...")
            print("-" * 70)
            result = await session.call_tool("create_model", {
                "model_id": "quadratic_model",
                "niterations": 20,
                "binary_operators": ["+", "-", "*", "/"],
                "unary_operators": ["square"],
                "population_size": 25,
                "populations": 10,
                "maxsize": 15
            })
            print(result.content[0].text)
            
            # Step 4: Check model status
            print("\n[4] Checking model status...")
            print("-" * 70)
            result = await session.call_tool("monitor_training", {
                "model_id": "quadratic_model"
            })
            print(result.content[0].text)
            
            # Step 5: Fit the model
            print("\n[5] Training model (this may take a moment)...")
            print("-" * 70)
            result = await session.call_tool("fit_model", {
                "model_id": "quadratic_model",
                "X_data": X_train.tolist(),
                "y_data": y_train.tolist(),
                "variable_names": ["x"]
            })
            print(result.content[0].text)
            
            # Step 6: Get discovered equations
            print("\n[6] Retrieving discovered equations...")
            print("-" * 70)
            result = await session.call_tool("get_equations", {
                "model_id": "quadratic_model",
                "n_equations": 5
            })
            print(result.content[0].text)
            
            # Step 7: Make predictions
            print("\n[7] Making predictions on test data...")
            print("-" * 70)
            result = await session.call_tool("predict", {
                "model_id": "quadratic_model",
                "X_new": X_test[:5].tolist()
            })
            print(result.content[0].text)
            
            # Step 8: Evaluate model
            print("\n[8] Evaluating model performance...")
            print("-" * 70)
            result = await session.call_tool("evaluate_model", {
                "model_id": "quadratic_model",
                "X_test": X_test.tolist(),
                "y_test": y_test.tolist()
            })
            print(result.content[0].text)
            
            # Step 9: Export best equation in different formats
            print("\n[9] Exporting best equation in different formats...")
            print("-" * 70)
            
            formats = ["sympy", "latex"]
            for fmt in formats:
                result = await session.call_tool("export_equation", {
                    "model_id": "quadratic_model",
                    "equation_index": 0,
                    "format": fmt
                })
                print(result.content[0].text)
            
            # Step 10: Save model
            print("\n[10] Saving model to disk...")
            print("-" * 70)
            result = await session.call_tool("save_model", {
                "model_id": "quadratic_model",
                "filepath": "quadratic_model.pkl"
            })
            print(result.content[0].text)
            
            # Step 11: Create a second model with different operators
            print("\n[11] Creating second model with different operators...")
            print("-" * 70)
            result = await session.call_tool("create_model", {
                "model_id": "trig_model",
                "niterations": 15,
                "binary_operators": ["+", "-", "*"],
                "unary_operators": ["sin", "cos"],
                "maxsize": 10
            })
            print(result.content[0].text)
            
            # Step 12: Update operators
            print("\n[12] Updating operators for the second model...")
            print("-" * 70)
            result = await session.call_tool("set_operators", {
                "model_id": "trig_model",
                "binary_operators": ["+", "*"],
                "unary_operators": ["sin", "cos", "exp"]
            })
            print(result.content[0].text)
            
            # Step 13: List all models
            print("\n[13] Reading models resource...")
            print("-" * 70)
            resource_content = await session.read_resource("models://list")
            print(resource_content.contents[0].text)
            
            # Step 14: Load saved model with new ID
            print("\n[14] Loading saved model with new ID...")
            print("-" * 70)
            result = await session.call_tool("load_model", {
                "model_id": "loaded_model",
                "filepath": "quadratic_model.pkl"
            })
            print(result.content[0].text)
            
            # Step 15: Make predictions with loaded model
            print("\n[15] Making predictions with loaded model...")
            print("-" * 70)
            result = await session.call_tool("predict", {
                "model_id": "loaded_model",
                "X_new": [[1.0], [2.0], [3.0]]
            })
            print(result.content[0].text)
            
            # Step 16: Final model status check
            print("\n[16] Final status check for all models...")
            print("-" * 70)
            for model_id in ["quadratic_model", "trig_model", "loaded_model"]:
                result = await session.call_tool("monitor_training", {
                    "model_id": model_id
                })
                print(result.content[0].text)
                print()
            
            print("=" * 70)
            print("✅ All demonstrations completed successfully!")
            print("=" * 70)
            
            print("\n💡 What you can do next:")
            print("  • Try different operators and parameters")
            print("  • Experiment with your own datasets")
            print("  • Compare multiple models on the same data")
            print("  • Export equations for use in other applications")
            print("  • Fine-tune hyperparameters for better results")

if __name__ == "__main__":
    asyncio.run(main())