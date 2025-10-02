from fastmcp import FastMCP
from pysr import PySRRegressor
import numpy as np
import pandas as pd
import pickle
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
import os

# Create FastMCP server instance
mcp = FastMCP("PySR Symbolic Regression Server")

# Global storage for models
models: Dict[str, PySRRegressor] = {}
training_status: Dict[str, Dict[str, Any]] = {}
model_data: Dict[str, Dict[str, np.ndarray]] = {}


@mcp.tool()
def create_model(
    model_id: str,
    niterations: int = 40,
    binary_operators: List[str] = ["+", "-", "*", "/"],
    unary_operators: List[str] = ["sin", "cos", "exp", "log"],
    population_size: int = 33,
    populations: int = 15,
    maxsize: int = 20,
    ncycles_per_iteration: int = 550
) -> str:
    """
    Create a new PySR model with specified parameters.
    
    Args:
        model_id: Unique identifier for the model
        niterations: Number of iterations for the evolutionary algorithm
        binary_operators: List of binary operators to use (e.g., ["+", "-", "*", "/"])
        unary_operators: List of unary operators to use (e.g., ["sin", "cos", "exp"])
        population_size: Number of individuals in each population
        populations: Number of populations to evolve
        maxsize: Maximum complexity of equations
        ncycles_per_iteration: Number of cycles per iteration
    
    Returns:
        Confirmation message with model details
    """
    try:
        model = PySRRegressor(
            niterations=niterations,
            binary_operators=binary_operators,
            unary_operators=unary_operators,
            population_size=population_size,
            populations=populations,
            maxsize=maxsize,
            ncycles_per_iteration=ncycles_per_iteration,
            progress=False,
            verbosity=0
        )
        
        models[model_id] = model
        training_status[model_id] = {
            "status": "created",
            "created_at": datetime.now().isoformat(),
            "fitted": False
        }
        
        return f"""Model '{model_id}' created successfully!
Configuration:
- Iterations: {niterations}
- Binary operators: {', '.join(binary_operators)}
- Unary operators: {', '.join(unary_operators)}
- Population size: {population_size}
- Populations: {populations}
- Max equation size: {maxsize}"""
    
    except Exception as e:
        return f"Error creating model: {str(e)}"


@mcp.tool()
def fit_model(
    model_id: str,
    X_data: List[List[float]],
    y_data: List[float],
    variable_names: Optional[List[str]] = None
) -> str:
    """
    Fit a PySR model on the provided data.
    
    Args:
        model_id: ID of the model to fit
        X_data: Input features as list of lists (each inner list is a sample)
        y_data: Target values as list
        variable_names: Optional names for input variables (e.g., ["x", "y", "z"])
    
    Returns:
        Training results and best equations found
    """
    try:
        if model_id not in models:
            return f"Error: Model '{model_id}' not found. Create it first using create_model."
        
        # Convert to numpy arrays
        X = np.array(X_data)
        y = np.array(y_data)
        
        # Store data for later predictions
        model_data[model_id] = {"X": X, "y": y}
        
        # Update training status
        training_status[model_id]["status"] = "training"
        training_status[model_id]["started_at"] = datetime.now().isoformat()
        
        # Fit the model
        model = models[model_id]
        model.fit(X, y, variable_names=variable_names)
        
        # Update training status
        training_status[model_id]["status"] = "completed"
        training_status[model_id]["completed_at"] = datetime.now().isoformat()
        training_status[model_id]["fitted"] = True
        
        # Get best equations
        equations = model.get_best()
        
        result = f"""Model '{model_id}' trained successfully!

Data shape: X={X.shape}, y={y.shape}

Top 3 Equations Found:
"""
        
        # Show top 3 equations
        for i in range(min(3, len(model.equations_))):
            eq = model.equations_.iloc[i]
            result += f"\n{i+1}. Complexity: {eq['complexity']}, Loss: {eq['loss']:.6f}"
            result += f"\n   Equation: {eq['equation']}\n"
        
        return result
    
    except Exception as e:
        training_status[model_id]["status"] = "failed"
        training_status[model_id]["error"] = str(e)
        return f"Error fitting model: {str(e)}"


@mcp.tool()
def predict(
    model_id: str,
    X_new: List[List[float]]
) -> str:
    """
    Make predictions using a trained model.
    
    Args:
        model_id: ID of the trained model
        X_new: New input data for predictions
    
    Returns:
        Predictions as formatted string
    """
    try:
        if model_id not in models:
            return f"Error: Model '{model_id}' not found."
        
        if not training_status[model_id].get("fitted", False):
            return f"Error: Model '{model_id}' has not been fitted yet."
        
        model = models[model_id]
        X = np.array(X_new)
        predictions = model.predict(X)
        
        result = f"Predictions for model '{model_id}':\n\n"
        for i, (input_val, pred) in enumerate(zip(X_new, predictions)):
            result += f"Input {i+1}: {input_val} → Prediction: {pred:.6f}\n"
        
        return result
    
    except Exception as e:
        return f"Error making predictions: {str(e)}"


@mcp.tool()
def get_equations(
    model_id: str,
    n_equations: int = 5
) -> str:
    """
    Get the best equations discovered by the model.
    
    Args:
        model_id: ID of the trained model
        n_equations: Number of top equations to return
    
    Returns:
        Formatted list of equations with their metrics
    """
    try:
        if model_id not in models:
            return f"Error: Model '{model_id}' not found."
        
        if not training_status[model_id].get("fitted", False):
            return f"Error: Model '{model_id}' has not been fitted yet."
        
        model = models[model_id]
        equations_df = model.equations_
        
        result = f"Top {n_equations} Equations for model '{model_id}':\n\n"
        
        for i in range(min(n_equations, len(equations_df))):
            eq = equations_df.iloc[i]
            result += f"Equation #{i+1}:\n"
            result += f"  Formula: {eq['equation']}\n"
            result += f"  Complexity: {eq['complexity']}\n"
            result += f"  Loss: {eq['loss']:.8f}\n"
            result += f"  Score: {eq['score']:.8f}\n\n"
        
        return result
    
    except Exception as e:
        return f"Error retrieving equations: {str(e)}"


@mcp.tool()
def save_model(
    model_id: str,
    filepath: str
) -> str:
    """
    Save a trained model to disk.
    
    Args:
        model_id: ID of the model to save
        filepath: Path where to save the model (should end with .pkl)
    
    Returns:
        Confirmation message
    """
    try:
        if model_id not in models:
            return f"Error: Model '{model_id}' not found."
        
        model = models[model_id]
        
        # Save model using pickle
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        # Also save metadata
        metadata = {
            "model_id": model_id,
            "training_status": training_status.get(model_id, {}),
            "saved_at": datetime.now().isoformat()
        }
        
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return f"""Model '{model_id}' saved successfully!
- Model file: {filepath}
- Metadata file: {metadata_path}
- File size: {os.path.getsize(filepath)} bytes"""
    
    except Exception as e:
        return f"Error saving model: {str(e)}"


@mcp.tool()
def load_model(
    model_id: str,
    filepath: str
) -> str:
    """
    Load a saved model from disk.
    
    Args:
        model_id: ID to assign to the loaded model
        filepath: Path to the saved model file
    
    Returns:
        Confirmation message with model details
    """
    try:
        # Load model
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        models[model_id] = model
        
        # Load metadata if available
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                training_status[model_id] = metadata.get("training_status", {})
        else:
            training_status[model_id] = {
                "status": "loaded",
                "fitted": True,
                "loaded_at": datetime.now().isoformat()
            }
        
        return f"""Model loaded successfully as '{model_id}'!
- Source file: {filepath}
- Status: {training_status[model_id].get('status', 'unknown')}
- Equations available: {len(model.equations_) if hasattr(model, 'equations_') else 0}"""
    
    except Exception as e:
        return f"Error loading model: {str(e)}"


@mcp.tool()
def export_equation(
    model_id: str,
    equation_index: int = 0,
    format: str = "sympy"
) -> str:
    """
    Export a specific equation in various formats.
    
    Args:
        model_id: ID of the trained model
        equation_index: Index of the equation to export (0 = best)
        format: Export format ("sympy", "latex", "numpy", "torch", "jax")
    
    Returns:
        Equation in the requested format
    """
    try:
        if model_id not in models:
            return f"Error: Model '{model_id}' not found."
        
        if not training_status[model_id].get("fitted", False):
            return f"Error: Model '{model_id}' has not been fitted yet."
        
        model = models[model_id]
        
        if equation_index >= len(model.equations_):
            return f"Error: Equation index {equation_index} out of range. Model has {len(model.equations_)} equations."
        
        result = f"Equation #{equation_index} from model '{model_id}':\n\n"
        
        if format == "sympy":
            eq = model.sympy(equation_index)
            result += f"SymPy format:\n{eq}\n"
        elif format == "latex":
            eq = model.latex(equation_index)
            result += f"LaTeX format:\n{eq}\n"
        elif format == "numpy":
            # Get the equation string
            eq_str = model.equations_.iloc[equation_index]['equation']
            result += f"NumPy format:\n{eq_str}\n"
            result += f"\nNote: Use model.predict() for numpy-based predictions\n"
        elif format == "torch":
            eq = model.pytorch(equation_index)
            result += f"PyTorch format:\n{eq}\n"
        elif format == "jax":
            eq = model.jax(equation_index)
            result += f"JAX format:\n{eq}\n"
        else:
            return f"Error: Unknown format '{format}'. Use: sympy, latex, numpy, torch, or jax"
        
        return result
    
    except Exception as e:
        return f"Error exporting equation: {str(e)}"


@mcp.tool()
def set_operators(
    model_id: str,
    binary_operators: Optional[List[str]] = None,
    unary_operators: Optional[List[str]] = None
) -> str:
    """
    Update the operators for an existing model (before fitting).
    
    Args:
        model_id: ID of the model to update
        binary_operators: New list of binary operators
        unary_operators: New list of unary operators
    
    Returns:
        Confirmation message
    """
    try:
        if model_id not in models:
            return f"Error: Model '{model_id}' not found."
        
        if training_status[model_id].get("fitted", False):
            return f"Warning: Model '{model_id}' has already been fitted. Create a new model to change operators."
        
        model = models[model_id]
        
        if binary_operators is not None:
            model.binary_operators = binary_operators
        
        if unary_operators is not None:
            model.unary_operators = unary_operators
        
        return f"""Operators updated for model '{model_id}':
- Binary operators: {', '.join(model.binary_operators)}
- Unary operators: {', '.join(model.unary_operators)}"""
    
    except Exception as e:
        return f"Error setting operators: {str(e)}"


@mcp.tool()
def validate_data(
    X_data: List[List[float]],
    y_data: List[float]
) -> str:
    """
    Validate input data for training.
    
    Args:
        X_data: Input features
        y_data: Target values
    
    Returns:
        Validation results and data statistics
    """
    try:
        X = np.array(X_data)
        y = np.array(y_data)
        
        issues = []
        warnings = []
        
        # Check dimensions
        if len(X) != len(y):
            issues.append(f"Dimension mismatch: X has {len(X)} samples, y has {len(y)} samples")
        
        # Check for NaN or Inf
        if np.isnan(X).any():
            issues.append("X contains NaN values")
        if np.isnan(y).any():
            issues.append("y contains NaN values")
        if np.isinf(X).any():
            issues.append("X contains infinite values")
        if np.isinf(y).any():
            issues.append("y contains infinite values")
        
        # Check for sufficient data
        if len(X) < 10:
            warnings.append(f"Small dataset: only {len(X)} samples (recommend at least 10)")
        
        # Get statistics
        result = "Data Validation Results:\n\n"
        
        if issues:
            result += "❌ ISSUES FOUND:\n"
            for issue in issues:
                result += f"  - {issue}\n"
            result += "\n"
        
        if warnings:
            result += "⚠️  WARNINGS:\n"
            for warning in warnings:
                result += f"  - {warning}\n"
            result += "\n"
        
        if not issues:
            result += "✅ Data is valid for training!\n\n"
        
        result += f"""Statistics:
- Number of samples: {len(X)}
- Number of features: {X.shape[1] if len(X.shape) > 1 else 1}
- Target range: [{y.min():.4f}, {y.max():.4f}]
- Target mean: {y.mean():.4f}
- Target std: {y.std():.4f}"""
        
        return result
    
    except Exception as e:
        return f"Error validating data: {str(e)}"


@mcp.tool()
def monitor_training(
    model_id: str
) -> str:
    """
    Get the current training status and progress of a model.
    
    Args:
        model_id: ID of the model to monitor
    
    Returns:
        Training status and progress information
    """
    try:
        if model_id not in models:
            return f"Error: Model '{model_id}' not found."
        
        status = training_status.get(model_id, {})
        
        result = f"Training Status for model '{model_id}':\n\n"
        result += f"Status: {status.get('status', 'unknown')}\n"
        result += f"Fitted: {status.get('fitted', False)}\n"
        
        if "created_at" in status:
            result += f"Created: {status['created_at']}\n"
        if "started_at" in status:
            result += f"Started: {status['started_at']}\n"
        if "completed_at" in status:
            result += f"Completed: {status['completed_at']}\n"
        
        if status.get("fitted", False):
            model = models[model_id]
            if hasattr(model, 'equations_'):
                result += f"\nEquations discovered: {len(model.equations_)}\n"
                best = model.equations_.iloc[0]
                result += f"Best equation loss: {best['loss']:.8f}\n"
                result += f"Best equation complexity: {best['complexity']}\n"
        
        if "error" in status:
            result += f"\n❌ Error: {status['error']}\n"
        
        return result
    
    except Exception as e:
        return f"Error monitoring training: {str(e)}"


@mcp.tool()
def evaluate_model(
    model_id: str,
    X_test: Optional[List[List[float]]] = None,
    y_test: Optional[List[float]] = None
) -> str:
    """
    Evaluate model performance on test data or training data.
    
    Args:
        model_id: ID of the trained model
        X_test: Test features (optional, uses training data if not provided)
        y_test: Test targets (optional, uses training data if not provided)
    
    Returns:
        Evaluation metrics (MSE, RMSE, R², MAE)
    """
    try:
        if model_id not in models:
            return f"Error: Model '{model_id}' not found."
        
        if not training_status[model_id].get("fitted", False):
            return f"Error: Model '{model_id}' has not been fitted yet."
        
        model = models[model_id]
        
        # Use test data if provided, otherwise use training data
        if X_test is not None and y_test is not None:
            X = np.array(X_test)
            y = np.array(y_test)
            data_type = "test"
        elif model_id in model_data:
            X = model_data[model_id]["X"]
            y = model_data[model_id]["y"]
            data_type = "training"
        else:
            return "Error: No test data provided and no training data available."
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - y_pred))
        
        # R² score
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        result = f"""Model Evaluation for '{model_id}' on {data_type} data:

Metrics:
- Mean Squared Error (MSE): {mse:.8f}
- Root Mean Squared Error (RMSE): {rmse:.8f}
- Mean Absolute Error (MAE): {mae:.8f}
- R² Score: {r2:.8f}

Data size: {len(X)} samples

Best Equation: {model.equations_.iloc[0]['equation']}
Equation Complexity: {model.equations_.iloc[0]['complexity']}
"""
        
        return result
    
    except Exception as e:
        return f"Error evaluating model: {str(e)}"


@mcp.resource("models://list")
def list_models() -> str:
    """Get a list of all available models"""
    if not models:
        return "No models created yet."
    
    result = "Available Models:\n\n"
    for model_id, status in training_status.items():
        result += f"📊 {model_id}\n"
        result += f"   Status: {status.get('status', 'unknown')}\n"
        result += f"   Fitted: {status.get('fitted', False)}\n"
        if status.get('fitted') and model_id in models:
            model = models[model_id]
            if hasattr(model, 'equations_'):
                result += f"   Equations: {len(model.equations_)}\n"
        result += "\n"
    
    return result


if __name__ == "__main__":
    mcp.run()