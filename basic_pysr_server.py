"""
Basic PySR MCP Server with FastMCP
Minimal version with one PySR tool
"""

from fastmcp import FastMCP
import numpy as np
import json

mcp = FastMCP("Basic PySR Server")


@mcp.tool()
def symbolic_regression(
    x_data: str,
    y_data: str,
    niterations: int = 3
) -> str:
    """
    Perform symbolic regression to discover mathematical equations.
    
    Args:
        x_data: JSON string of x values (e.g., "[-2, -1, 0, 1, 2]")
        y_data: JSON string of y values (e.g., "[4, 1, 0, 1, 4]")
        niterations: Number of iterations (default: 3, recommend keeping low for speed)
    
    Returns:
        JSON string with discovered equation and details
    
    Note: First run may take 2-3 minutes due to Julia compilation.
          Subsequent runs are faster (30-60 seconds).
    """
    try:
        from pysr import PySRRegressor
        
        # Parse data
        X = np.array(json.loads(x_data))
        y = np.array(json.loads(y_data))
        
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Create model with MINIMAL settings for speed
        model = PySRRegressor(
            niterations=niterations,
            binary_operators=["+", "-", "*"],  # Only basic operators
            unary_operators=[],  # No unary operators for speed
            population_size=10,  # Small population
            maxsize=8,  # Low complexity limit
            verbosity=1,  # Show progress
            procs=1,  # Single process
            multithreading=False,
            random_state=42,
            timeout_in_seconds=120  # 2 minute timeout
        )
        
        print(f"Starting PySR with {len(X)} samples, {niterations} iterations...")
        
        # Fit the model
        model.fit(X, y)
        
        # Get results
        best_eq = str(model.sympy())
        equations = model.equations_
        
        result = {
            "success": True,
            "best_equation": best_eq,
            "n_equations": len(equations),
            "top_equations": []
        }
        
        # Add top 3 equations
        for i in range(min(3, len(equations))):
            row = equations.iloc[i]
            result["top_equations"].append({
                "equation": str(row["equation"]),
                "complexity": int(row["complexity"]),
                "loss": float(row["loss"])
            })
        
        return json.dumps(result, indent=2)
        
    except ImportError:
        return json.dumps({
            "success": False,
            "error": "PySR not installed. Run: pip install pysr && python -c 'import pysr; pysr.install()'"
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"PySR error: {str(e)}"
        })


@mcp.tool()
def create_test_data(
    equation_type: str,
    n_points: int = 20
) -> str:
    """
    Generate test data for symbolic regression.
    
    Args:
        equation_type: Type of equation ("linear", "quadratic", "cubic")
        n_points: Number of data points (default: 20, keep small for speed)
    
    Returns:
        JSON string with x and y data
    """
    try:
        X = np.linspace(-3, 3, n_points)
        
        if equation_type == "linear":
            y = 2 * X + 1
            true_eq = "2*x + 1"
        elif equation_type == "quadratic":
            y = X**2 + X + 1
            true_eq = "x^2 + x + 1"
        elif equation_type == "cubic":
            y = X**3 - X + 1
            true_eq = "x^3 - x + 1"
        else:
            return json.dumps({
                "success": False,
                "error": f"Unknown equation type: {equation_type}"
            })
        
        # Add small noise
        y += np.random.normal(0, 0.1, y.shape)
        
        return json.dumps({
            "success": True,
            "x_data": X.tolist(),
            "y_data": y.tolist(),
            "true_equation": true_eq,
            "n_points": n_points
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


if __name__ == "__main__":
    print("=" * 60)
    print("Basic PySR MCP Server Starting...")
    print("=" * 60)
    print("Note: First PySR run will be slow (Julia compilation)")
    print("=" * 60)
    mcp.run()