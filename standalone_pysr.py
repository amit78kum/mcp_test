"""
Standalone PySR Equation Discovery
No MCP server required - works directly!
"""

from pysr import PySRRegressor
import numpy as np


def train_and_get_best_equation(
    X_data,
    y_data,
    niterations=10,
    variable_names=None
):
    """
    Train PySR and return the best equation.
    
    Args:
        X_data: Input features (list of lists or numpy array)
        y_data: Target values (list or numpy array)
        niterations: Number of training iterations
        variable_names: Optional variable names
    
    Returns:
        Dictionary with equation info
    """
    
    # Convert to numpy
    X = np.array(X_data)
    y = np.array(y_data)
    
    print(f"📊 Training on {len(X)} samples...")
    print(f"   Features: {X.shape[1] if len(X.shape) > 1 else 1}")
    print(f"   Iterations: {niterations}")
    print(f"   This may take 30-60 seconds on first run...\n")
    
    # Create model
    model = PySRRegressor(
        niterations=niterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "cube", "sqrt", "sin", "cos", "exp", "log"],
        population_size=30,
        populations=15,
        maxsize=20,
        progress=True,  # Show progress
        verbosity=0,
        timeout_in_seconds=300
    )
    
    # Train
    model.fit(X, y, variable_names=variable_names)
    
    # Get best equation
    best = model.equations_.iloc[0]
    
    result = {
        "equation": best['equation'],
        "complexity": int(best['complexity']),
        "loss": float(best['loss']),
        "score": float(best['score']),
        "model": model,
        "all_equations": model.equations_
    }
    
    return result


def print_result(result):
    """Pretty print the result"""
    print("\n" + "=" * 70)
    print("✅ EQUATION DISCOVERED!")
    print("=" * 70)
    print(f"\n📝 Best Equation: {result['equation']}")
    print(f"\n📊 Metrics:")
    print(f"   • Complexity: {result['complexity']}")
    print(f"   • Loss: {result['loss']:.8f}")
    print(f"   • Score: {result['score']:.4f}")
    print(f"\n📋 Total equations found: {len(result['all_equations'])}")
    print("=" * 70 + "\n")


def show_top_equations(result, n=5):
    """Show top N equations"""
    print(f"\n📊 Top {n} Equations:")
    print("-" * 70)
    
    equations = result['all_equations']
    for i in range(min(n, len(equations))):
        eq = equations.iloc[i]
        print(f"\n#{i+1}")
        print(f"  Equation:   {eq['equation']}")
        print(f"  Complexity: {eq['complexity']}")
        print(f"  Loss:       {eq['loss']:.8f}")
        print(f"  Score:      {eq['score']:.4f}")
    
    print("\n" + "-" * 70)


def make_predictions(result, X_new):
    """Make predictions with the best equation"""
    model = result['model']
    X = np.array(X_new)
    predictions = model.predict(X)
    
    print("\n🔮 Predictions:")
    print("-" * 70)
    for i, (input_val, pred) in enumerate(zip(X_new, predictions)):
        print(f"  Input: {input_val} → Prediction: {pred:.6f}")
    print("-" * 70 + "\n")
    
    return predictions


def export_equation(result, format="latex"):
    """Export equation in different formats"""
    model = result['model']
    
    print(f"\n📤 Export Format: {format.upper()}")
    print("-" * 70)
    
    if format == "latex":
        latex = model.latex(0)
        print(f"LaTeX: {latex}")
        print(f"\nUse in documents: $${latex}$$")
        
    elif format == "sympy":
        sympy_eq = model.sympy(0)
        print(f"SymPy: {sympy_eq}")
        
    elif format == "python":
        eq_str = result['equation']
        n_vars = model.n_features_in_
        var_list = ', '.join([f'x{i}' for i in range(n_vars)])
        print(f"def equation({var_list}):")
        print(f"    return {eq_str}")
    
    print("-" * 70 + "\n")


# ============================================================================
# EXAMPLES
# ============================================================================

def example1_linear():
    """Example 1: Linear equation y = 2x + 3"""
    print("\n" + "=" * 70)
    print("Example 1: Linear Equation (y = 2x + 3)")
    print("=" * 70)
    
    X = [[1], [2], [3], [4], [5]]
    y = [5, 7, 9, 11, 13]
    
    result = train_and_get_best_equation(
        X_data=X,
        y_data=y,
        niterations=5,
        variable_names=["x"]
    )
    
    print_result(result)
    
    # Test predictions
    print("Testing predictions:")
    X_test = [[6], [7], [8]]
    predictions = make_predictions(result, X_test)
    
    print("Expected (y = 2x + 3):")
    for x in X_test:
        expected = 2 * x[0] + 3
        print(f"  x={x[0]} → y={expected}")


def example2_quadratic():
    """Example 2: Quadratic equation y = x² + 1"""
    print("\n" + "=" * 70)
    print("Example 2: Quadratic Equation (y = x² + 1)")
    print("=" * 70)
    
    X = [[-2], [-1], [0], [1], [2], [3], [4]]
    y = [5, 2, 1, 2, 5, 10, 17]
    
    result = train_and_get_best_equation(
        X_data=X,
        y_data=y,
        niterations=10,
        variable_names=["x"]
    )
    
    print_result(result)
    show_top_equations(result, n=3)
    
    # Export
    export_equation(result, format="latex")
    export_equation(result, format="python")


def example3_multivariable():
    """Example 3: Multi-variable z = x² + y²"""
    print("\n" + "=" * 70)
    print("Example 3: Multi-Variable Equation (z = x² + y²)")
    print("=" * 70)
    
    np.random.seed(42)
    X = np.random.uniform(-3, 3, (30, 2))
    y = X[:, 0]**2 + X[:, 1]**2
    
    result = train_and_get_best_equation(
        X_data=X.tolist(),
        y_data=y.tolist(),
        niterations=10,
        variable_names=["x", "y"]
    )
    
    print_result(result)
    
    # Test predictions
    print("Testing predictions:")
    X_test = [[1, 0], [0, 1], [1, 1], [2, 2]]
    predictions = make_predictions(result, X_test)
    
    print("Expected (z = x² + y²):")
    for x, y_val in X_test:
        expected = x**2 + y_val**2
        print(f"  x={x}, y={y_val} → z={expected}")


def custom_example():
    """Let user input their own data"""
    print("\n" + "=" * 70)
    print("Custom Example - Your Data")
    print("=" * 70)
    
    print("\nEnter your data:")
    print("(Simple format: space-separated X values and Y values)")
    print("\nExample for y = x²:")
    print("  X: 1 2 3 4 5")
    print("  Y: 1 4 9 16 25")
    
    try:
        x_input = input("\nEnter X values (space-separated): ")
        y_input = input("Enter Y values (space-separated): ")
        
        X = [[float(x)] for x in x_input.split()]
        y = [float(y_val) for y_val in y_input.split()]
        
        if len(X) != len(y):
            print("\n❌ Error: X and Y must have same length!")
            return
        
        if len(X) < 5:
            print("\n⚠️ Warning: You should have at least 5 data points!")
        
        var_name = input("Variable name (default: x): ").strip() or "x"
        iterations = input("Iterations (default: 10): ").strip()
        iterations = int(iterations) if iterations else 10
        
        result = train_and_get_best_equation(
            X_data=X,
            y_data=y,
            niterations=iterations,
            variable_names=[var_name]
        )
        
        print_result(result)
        show_top_equations(result, n=3)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main menu"""
    print("\n" + "=" * 70)
    print("🔬 PySR Standalone - Equation Discovery")
    print("=" * 70)
    print("\nThis works directly without MCP server!")
    print("\nChoose an example:")
    print("  1. Linear equation (y = 2x + 3)")
    print("  2. Quadratic equation (y = x² + 1)")
    print("  3. Multi-variable (z = x² + y²)")
    print("  4. Custom data (enter your own)")
    print("  5. Run all examples")
    print()
    
    choice = input("Enter choice (1-5, default=1): ").strip() or "1"
    
    try:
        if choice == "1":
            example1_linear()
        elif choice == "2":
            example2_quadratic()
        elif choice == "3":
            example3_multivariable()
        elif choice == "4":
            custom_example()
        elif choice == "5":
            example1_linear()
            input("\nPress Enter to continue to example 2...")
            example2_quadratic()
            input("\nPress Enter to continue to example 3...")
            example3_multivariable()
        else:
            print("Invalid choice!")
            return
        
        print("\n" + "=" * 70)
        print("✅ Done!")
        print("=" * 70)
        print("\n💡 This standalone version works great!")
        print("   If you need MCP server features, run:")
        print("   python test_server_startup.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")