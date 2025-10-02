from fastmcp import FastMCP
from pysr import PySRRegressor
import numpy as np
from typing import List, Optional

# Create FastMCP server
mcp = FastMCP("PySR Best Equation Server")

# Global storage
models = {}


@mcp.tool()
def train_and_get_best_equation(
    X_data: List[List[float]],
    y_data: List[float],
    model_id: str = "default",
    niterations: int = 20,
    variable_names: Optional[List[str]] = None
) -> str:
    """
    Train a PySR model and return the best equation discovered.
    
    Args:
        X_data: Input features as list of lists
        y_data: Target values as list
        model_id: Unique identifier for this model
        niterations: Number of iterations for training
        variable_names: Optional names for variables (e.g., ["x", "y"])
    
    Returns:
        The best equation found with its metrics
    """
    try:
        # Convert to numpy
        X = np.array(X_data)
        y = np.array(y_data)
        
        # Create and train model
        model = PySRRegressor(
            niterations=niterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["square", "cube", "sqrt", "sin", "cos", "exp", "log"],
            population_size=30,
            populations=15,
            maxsize=20,
            progress=False,
            verbosity=0,
            timeout_in_seconds=300
        )
        
        print(f"Training model '{model_id}'...")
        model.fit(X, y, variable_names=variable_names)
        
        # Store model
        models[model_id] = model
        
        # Get best equation
        best = model.equations_.iloc[0]
        
        result = f"""✅ Model '{model_id}' trained successfully!

📊 BEST EQUATION FOUND:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Equation: {best['equation']}

Metrics:
  • Complexity: {best['complexity']}
  • Loss: {best['loss']:.8f}
  • Score: {best['score']:.8f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Data Info:
  • Training samples: {len(X)}
  • Features: {X.shape[1] if len(X.shape) > 1 else 1}
  • Total equations discovered: {len(model.equations_)}
"""
        
        return result
        
    except Exception as e:
        return f"❌ Error: {str(e)}"


@mcp.tool()
def get_best_equation(
    model_id: str = "default"
) -> str:
    """
    Get the best equation from a previously trained model.
    
    Args:
        model_id: ID of the trained model
    
    Returns:
        The best equation with its details
    """
    try:
        if model_id not in models:
            return f"❌ Error: Model '{model_id}' not found. Train a model first."
        
        model = models[model_id]
        
        if not hasattr(model, 'equations_'):
            return f"❌ Error: Model '{model_id}' has not been trained yet."
        
        # Get best equation
        best = model.equations_.iloc[0]
        
        result = f"""📊 BEST EQUATION for model '{model_id}':
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Equation: {best['equation']}

Metrics:
  • Complexity: {best['complexity']}
  • Loss: {best['loss']:.8f}
  • Score: {best['score']:.8f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        return result
        
    except Exception as e:
        return f"❌ Error: {str(e)}"


@mcp.tool()
def get_top_equations(
    model_id: str = "default",
    n: int = 5
) -> str:
    """
    Get the top N equations from a trained model.
    
    Args:
        model_id: ID of the trained model
        n: Number of top equations to return
    
    Returns:
        Top N equations with their metrics
    """
    try:
        if model_id not in models:
            return f"❌ Error: Model '{model_id}' not found."
        
        model = models[model_id]
        
        if not hasattr(model, 'equations_'):
            return f"❌ Error: Model '{model_id}' has not been trained yet."
        
        equations = model.equations_
        n = min(n, len(equations))
        
        result = f"""📊 TOP {n} EQUATIONS for model '{model_id}':
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
        
        for i in range(n):
            eq = equations.iloc[i]
            result += f"""
#{i+1} - Equation: {eq['equation']}
     Complexity: {eq['complexity']} | Loss: {eq['loss']:.8f} | Score: {eq['score']:.8f}
"""
        
        result += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error: {str(e)}"


@mcp.tool()
def predict_with_best_equation(
    model_id: str,
    X_new: List[List[float]]
) -> str:
    """
    Make predictions using the best equation from a trained model.
    
    Args:
        model_id: ID of the trained model
        X_new: New input data for predictions
    
    Returns:
        Predictions using the best equation
    """
    try:
        if model_id not in models:
            return f"❌ Error: Model '{model_id}' not found."
        
        model = models[model_id]
        
        if not hasattr(model, 'equations_'):
            return f"❌ Error: Model '{model_id}' has not been trained yet."
        
        X = np.array(X_new)
        predictions = model.predict(X)
        
        best_eq = model.equations_.iloc[0]['equation']
        
        result = f"""🔮 PREDICTIONS using best equation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model: {model_id}
Equation: {best_eq}

Predictions:
"""
        
        for i, (input_val, pred) in enumerate(zip(X_new, predictions)):
            result += f"  {i+1}. Input: {input_val} → Prediction: {pred:.6f}\n"
        
        result += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error: {str(e)}"


@mcp.tool()
def export_best_equation(
    model_id: str = "default",
    format: str = "latex"
) -> str:
    """
    Export the best equation in various formats.
    
    Args:
        model_id: ID of the trained model
        format: Export format ("latex", "sympy", "python")
    
    Returns:
        The best equation in the requested format
    """
    try:
        if model_id not in models:
            return f"❌ Error: Model '{model_id}' not found."
        
        model = models[model_id]
        
        if not hasattr(model, 'equations_'):
            return f"❌ Error: Model '{model_id}' has not been trained yet."
        
        best_eq_str = model.equations_.iloc[0]['equation']
        
        result = f"""📤 BEST EQUATION EXPORT for model '{model_id}':
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Original: {best_eq_str}

"""
        
        if format == "latex":
            latex = model.latex(0)
            result += f"LaTeX:\n{latex}\n\n"
            result += "Use in LaTeX documents:\n"
            result += f"$${latex}$$\n"
            
        elif format == "sympy":
            sympy_eq = model.sympy(0)
            result += f"SymPy:\n{sympy_eq}\n\n"
            result += "Use in Python:\n"
            result += f"from sympy import *\nequation = {sympy_eq}\n"
            
        elif format == "python":
            result += f"Python function:\n"
            result += f"def equation({', '.join([f'x{i}' for i in range(model.n_features_in_)])}):\n"
            result += f"    return {best_eq_str}\n"
            
        else:
            result += f"❌ Unknown format '{format}'. Use: latex, sympy, or python\n"
        
        result += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error: {str(e)}"


@mcp.resource("models://list")
def list_all_models() -> str:
    """List all trained models"""
    if not models:
        return "No models trained yet."
    
    result = "📋 TRAINED MODELS:\n\n"
    
    for model_id, model in models.items():
        result += f"• {model_id}\n"
        if hasattr(model, 'equations_'):
            best = model.equations_.iloc[0]
            result += f"  Best equation: {best['equation']}\n"
            result += f"  Loss: {best['loss']:.8f}\n"
        result += "\n"
    
    return result


if __name__ == "__main__":
    print("🚀 Starting PySR MCP Server...")
    print("=" * 60)
    print("Available tools:")
    print("  1. train_and_get_best_equation - Train model and get best equation")
    print("  2. get_best_equation - Get best equation from trained model")
    print("  3. get_top_equations - Get top N equations")
    print("  4. predict_with_best_equation - Make predictions")
    print("  5. export_best_equation - Export in various formats")
    print("=" * 60)
    mcp.run()