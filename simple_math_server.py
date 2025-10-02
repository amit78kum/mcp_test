"""
Simple Math MCP Server using FastMCP (Works Instantly)
Fast alternative demonstrating MCP without slow PySR operations
"""

from fastmcp import FastMCP
import numpy as np
from typing import Optional
import json

# Initialize FastMCP server
mcp = FastMCP("Simple Math MCP Server")

@mcp.tool()
def polynomial_fit(
    x_data: str,
    y_data: str,
    degree: int = 2
) -> str:
    """
    Fit a polynomial to data using numpy (very fast).
    
    Args:
        x_data: JSON string of input data (list)
        y_data: JSON string of target values (list)
        degree: Polynomial degree (default: 2 for quadratic)
    
    Returns:
        JSON string with polynomial coefficients and equation
    """
    try:
        # Parse input data
        X = np.array(json.loads(x_data))
        y = np.array(json.loads(y_data))
        
        # Flatten if needed
        if X.ndim > 1:
            X = X.flatten()
        
        # Fit polynomial
        coefficients = np.polyfit(X, y, degree)
        
        # Create equation string
        terms = []
        for i, coef in enumerate(coefficients):
            power = degree - i
            if power == 0:
                terms.append(f"{coef:.4f}")
            elif power == 1:
                terms.append(f"{coef:.4f}*x")
            else:
                terms.append(f"{coef:.4f}*x^{power}")
        
        equation = " + ".join(terms).replace("+ -", "- ")
        
        # Calculate R-squared
        y_pred = np.polyval(coefficients, X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return json.dumps({
            "success": True,
            "equation": equation,
            "coefficients": coefficients.tolist(),
            "degree": degree,
            "r_squared": float(r_squared),
            "mean_squared_error": float(ss_res / len(y))
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Error during polynomial fit: {str(e)}"
        })


@mcp.tool()
def evaluate_expression(
    expression: str,
    x_data: str
) -> str:
    """
    Evaluate a mathematical expression with given x values.
    
    Args:
        expression: Math expression (e.g., "x**2 + 2*x + 1")
        x_data: JSON string of x values (list)
    
    Returns:
        JSON string with computed y values
    """
    try:
        # Parse input
        x_values = np.array(json.loads(x_data))
        
        # Flatten if needed
        if x_values.ndim > 1:
            x_values = x_values.flatten()
        
        # Safe evaluation
        allowed_names = {
            "x": x_values,
            "np": np,
            "sin": np.sin,
            "cos": np.cos,
            "exp": np.exp,
            "log": np.log,
            "sqrt": np.sqrt,
            "abs": np.abs
        }
        
        # Evaluate expression
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        
        return json.dumps({
            "success": True,
            "expression": expression,
            "x_values": x_values.tolist(),
            "y_values": result.tolist() if hasattr(result, 'tolist') else [float(result)]
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Error evaluating expression: {str(e)}"
        })


@mcp.tool()
def generate_sample_data(
    function: str,
    n_samples: int = 100,
    x_range: Optional[str] = None,
    noise_level: float = 0.0
) -> str:
    """
    Generate sample data for testing.
    
    Args:
        function: Function type ('linear', 'quadratic', 'cubic', 'sin', 'exp')
        n_samples: Number of samples to generate (default: 100)
        x_range: JSON string [min, max] for x values (default: [-10, 10])
        noise_level: Standard deviation of Gaussian noise (default: 0.0)
    
    Returns:
        JSON string with X and y data
    """
    try:
        # Parse range
        if x_range:
            x_min, x_max = json.loads(x_range)
        else:
            x_min, x_max = -10, 10
        
        # Generate X data
        X = np.linspace(x_min, x_max, n_samples)
        
        # Generate y based on function
        if function == 'linear':
            y = 2 * X + 1
            true_eq = "2*x + 1"
        elif function == 'quadratic':
            y = X**2 + 2*X + 1
            true_eq = "x^2 + 2*x + 1"
        elif function == 'cubic':
            y = X**3 - 2*X**2 + X + 1
            true_eq = "x^3 - 2*x^2 + x + 1"
        elif function == 'sin':
            y = np.sin(X)
            true_eq = "sin(x)"
        elif function == 'exp':
            y = np.exp(X / 5)
            true_eq = "exp(x/5)"
        else:
            return json.dumps({
                "success": False,
                "error": f"Unknown function: {function}"
            })
        
        # Add noise
        if noise_level > 0:
            y += np.random.normal(0, noise_level, y.shape)
        
        return json.dumps({
            "success": True,
            "X": X.tolist(),
            "y": y.tolist(),
            "function": function,
            "true_equation": true_eq,
            "n_samples": n_samples,
            "noise_level": noise_level
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Error generating data: {str(e)}"
        })


@mcp.tool()
def calculate_statistics(
    data: str
) -> str:
    """
    Calculate statistical measures for a dataset.
    
    Args:
        data: JSON string of numerical data (list)
    
    Returns:
        JSON string with statistics
    """
    try:
        values = np.array(json.loads(data))
        
        return json.dumps({
            "success": True,
            "count": int(len(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75))
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Error calculating statistics: {str(e)}"
        })


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()