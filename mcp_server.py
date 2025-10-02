"""
PySR MCP Server using FastMCP
Provides symbolic regression capabilities through MCP protocol
"""

from fastmcp import FastMCP
import numpy as np
from typing import List, Optional
import json

# Initialize FastMCP server
mcp = FastMCP("PySR Symbolic Regression Server")

@mcp.tool()
def fit_symbolic_regression(
    x_data: str,
    y_data: str,
    niterations: int = 40,
    binary_operators: Optional[str] = None,
    unary_operators: Optional[str] = None,
    population_size: int = 33,
    max_complexity: int = 10
) -> str:
    """
    Fit a symbolic regression model to find mathematical expressions.
    
    Args:
        x_data: JSON string of input data (list of lists for multiple features or single list)
        y_data: JSON string of target values (list)
        niterations: Number of iterations for evolution (default: 40)
        binary_operators: Comma-separated binary operators (default: "+,-,*,/")
        unary_operators: Comma-separated unary operators (default: "sin,cos,exp,log")
        population_size: Population size for genetic algorithm (default: 33)
        max_complexity: Maximum complexity of equations (default: 10)
    
    Returns:
        JSON string with best equations found
    """
    try:
        from pysr import PySRRegressor
        
        # Parse input data
        X = np.array(json.loads(x_data))
        y = np.array(json.loads(y_data))
        
        # Reshape X if it's 1D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Set default operators
        binary_ops = binary_operators.split(",") if binary_operators else ["+", "-", "*", "/"]
        unary_ops = unary_operators.split(",") if unary_operators else ["sin", "cos", "exp", "log"]
        
        # Initialize PySR model
        model = PySRRegressor(
            niterations=niterations,
            binary_operators=binary_ops,
            unary_operators=unary_ops,
            population_size=population_size,
            maxsize=max_complexity,
            verbosity=0,
            random_state=42
        )
        
        # Fit the model
        model.fit(X, y)
        
        # Get the equations
        equations = model.equations_
        
        # Format results
        results = {
            "best_equation": str(model.sympy()),
            "equations": [],
            "success": True
        }
        
        # Add top equations
        for idx, row in equations.head(5).iterrows():
            results["equations"].append({
                "complexity": int(row["complexity"]),
                "loss": float(row["loss"]),
                "equation": str(row["equation"]),
                "score": float(row.get("score", 0))
            })
        
        return json.dumps(results, indent=2)
        
    except ImportError:
        return json.dumps({
            "success": False,
            "error": "PySR is not installed. Install with: pip install pysr"
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Error during symbolic regression: {str(e)}"
        })


@mcp.tool()
def predict_with_equation(
    equation: str,
    x_data: str,
    feature_names: Optional[str] = None
) -> str:
    """
    Predict values using a symbolic equation.
    
    Args:
        equation: Mathematical equation as string (e.g., "x0**2 + sin(x1)")
        x_data: JSON string of input data (list of lists or single list)
        feature_names: Comma-separated feature names (default: "x0,x1,...")
    
    Returns:
        JSON string with predictions
    """
    try:
        import sympy as sp
        
        # Parse input data
        X = np.array(json.loads(x_data))
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Determine feature names
        n_features = X.shape[1]
        if feature_names:
            features = feature_names.split(",")
        else:
            features = [f"x{i}" for i in range(n_features)]
        
        # Create sympy symbols
        symbols = {name: sp.Symbol(name) for name in features}
        
        # Parse equation
        expr = sp.sympify(equation, locals=symbols)
        
        # Convert to numpy function
        func = sp.lambdify(list(symbols.values()), expr, "numpy")
        
        # Make predictions
        predictions = func(*[X[:, i] for i in range(n_features)])
        
        return json.dumps({
            "success": True,
            "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else [float(predictions)]
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Error during prediction: {str(e)}"
        })


@mcp.tool()
def generate_sample_data(
    function: str,
    n_samples: int = 100,
    x_range: Optional[str] = None,
    noise_level: float = 0.0
) -> str:
    """
    Generate sample data for testing symbolic regression.
    
    Args:
        function: Function type ('linear', 'quadratic', 'sin', 'exp', or custom equation)
        n_samples: Number of samples to generate (default: 100)
        x_range: JSON string [min, max] for x values (default: [-10, 10])
        noise_level: Standard deviation of Gaussian noise (default: 0.0)
    
    Returns:
        JSON string with X and y data
    """
    try:
        import sympy as sp
        
        # Parse range
        if x_range:
            x_min, x_max = json.loads(x_range)
        else:
            x_min, x_max = -10, 10
        
        # Generate X data
        X = np.linspace(x_min, x_max, n_samples).reshape(-1, 1)
        
        # Generate y based on function
        if function == 'linear':
            y = 2 * X[:, 0] + 1
        elif function == 'quadratic':
            y = X[:, 0]**2 + 2*X[:, 0] + 1
        elif function == 'sin':
            y = np.sin(X[:, 0])
        elif function == 'exp':
            y = np.exp(X[:, 0] / 5)
        else:
            # Try to parse as custom equation
            x0 = sp.Symbol('x0')
            expr = sp.sympify(function)
            func = sp.lambdify(x0, expr, "numpy")
            y = func(X[:, 0])
        
        # Add noise
        if noise_level > 0:
            y += np.random.normal(0, noise_level, y.shape)
        
        return json.dumps({
            "success": True,
            "X": X.tolist(),
            "y": y.tolist(),
            "function": function,
            "n_samples": n_samples
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Error generating data: {str(e)}"
        })


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()