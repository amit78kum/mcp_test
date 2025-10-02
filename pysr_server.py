"""
PySR MCP Server for Claude Desktop
"""

from fastmcp import FastMCP
import numpy as np
import json

mcp = FastMCP("PySR Symbolic Regression")


@mcp.tool()
def fit_equation(
    x_values: str,
    y_values: str,
    max_iterations: int = 5
) -> str:
    """
    Discover mathematical equations from data using symbolic regression.
    
    Args:
        x_values: Comma-separated x values (e.g., "1,2,3,4")
        y_values: Comma-separated y values (e.g., "1,4,9,16")
        max_iterations: Number of iterations (default: 5)
    
    Returns:
        The discovered equation
    """
    try:
        from pysr import PySRRegressor
        
        # Parse input
        try:
            X = np.array(json.loads(x_values))
        except:
            X = np.array([float(x.strip()) for x in x_values.split(',')])
        
        try:
            y = np.array(json.loads(y_values))
        except:
            y = np.array([float(y.strip()) for y in y_values.split(',')])
        
        if len(X) != len(y):
            return json.dumps({
                "success": False,
                "error": f"Length mismatch: {len(X)} x values, {len(y)} y values"
            })
        
        if len(X) < 5:
            return json.dumps({
                "success": False,
                "error": "Need at least 5 data points"
            })
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Configure PySR
        model = PySRRegressor(
            niterations=max_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["square"],
            population_size=15,
            maxsize=10,
            verbosity=0,
            procs=1,
            multithreading=False,
            random_state=42,
            timeout_in_seconds=90
        )
        
        # Fit
        model.fit(X, y)
        best_eq = str(model.sympy())
        
        return json.dumps({
            "success": True,
            "equation": best_eq,
            "message": f"Discovered equation: {best_eq}"
        }, indent=2)
        
    except ImportError:
        return json.dumps({
            "success": False,
            "error": "PySR not installed properly"
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


@mcp.tool()
def generate_test_data(
    pattern: str,
    num_points: int = 20
) -> str:
    """
    Generate sample data for testing.
    
    Args:
        pattern: "linear", "quadratic", "cubic", or "sine"
        num_points: Number of points (default: 20)
    
    Returns:
        Generated x and y values
    """
    try:
        X = np.linspace(-3, 3, num_points)
        
        if pattern == "linear":
            y = 2*X + 1
            eq = "2*x + 1"
        elif pattern == "quadratic":
            y = X**2 + X + 1
            eq = "x^2 + x + 1"
        elif pattern == "cubic":
            y = X**3 - 2*X
            eq = "x^3 - 2*x"
        elif pattern == "sine":
            y = np.sin(X)
            eq = "sin(x)"
        else:
            return json.dumps({
                "success": False,
                "error": "Pattern must be: linear, quadratic, cubic, or sine"
            })
        
        return json.dumps({
            "success": True,
            "x_values": X.tolist(),
            "y_values": y.tolist(),
            "true_equation": eq,
            "pattern": pattern
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


if __name__ == "__main__":
    mcp.run()