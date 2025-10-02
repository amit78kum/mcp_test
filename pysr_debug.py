"""
Debug script to verify PySR installation and setup
Run this first to diagnose any issues
"""

import sys
import subprocess

def check_python():
    """Check Python version"""
    print("🐍 Checking Python...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ❌ Python 3.8+ required")
        return False
    print("   ✅ Python version OK")
    return True


def check_julia():
    """Check if Julia is installed"""
    print("\n🔧 Checking Julia...")
    
    # Try multiple methods to find Julia (especially on Windows)
    julia_commands = ["julia", "julia.exe"]
    
    for cmd in julia_commands:
        try:
            result = subprocess.run(
                [cmd, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                shell=True  # Use shell on Windows
            )
            if result.returncode == 0:
                print(f"   {result.stdout.strip()}")
                print("   ✅ Julia is installed")
                return True
        except FileNotFoundError:
            continue
        except Exception:
            continue
    
    # Julia not found in PATH, but check if PySR can find it
    print("   ⚠️  Julia not found in system PATH")
    print("   ℹ️  Checking if PySR can find Julia...")
    
    try:
        # PySR has its own way of finding Julia
        import pysr
        # Try to create a minimal model - this will fail if Julia isn't available
        from pysr import PySRRegressor
        model = PySRRegressor(niterations=1)
        print("   ✅ Julia is available to PySR (even if not in PATH)")
        print("   ℹ️  This is fine - PySR can still work!")
        return True
    except Exception as e:
        print(f"   ❌ Julia not available to PySR either")
        print("\n   Installation instructions:")
        print("   • macOS: brew install julia")
        print("   • Ubuntu: sudo snap install julia --classic")
        print("   • Windows: Download from https://julialang.org/downloads/")
        return False


def check_numpy():
    """Check NumPy installation"""
    print("\n📊 Checking NumPy...")
    try:
        import numpy as np
        print(f"   NumPy version: {np.__version__}")
        print("   ✅ NumPy installed")
        return True
    except ImportError:
        print("   ❌ NumPy not installed")
        print("   Install with: pip install numpy")
        return False


def check_pysr():
    """Check PySR installation"""
    print("\n🔬 Checking PySR...")
    try:
        import pysr
        print(f"   PySR version: {pysr.__version__}")
        print("   ✅ PySR package installed")
        return True
    except ImportError:
        print("   ❌ PySR not installed")
        print("   Install with: pip install pysr")
        return False


def test_pysr_basic():
    """Test basic PySR functionality"""
    print("\n🧪 Testing PySR basic functionality...")
    try:
        from pysr import PySRRegressor
        import numpy as np
        
        print("   Creating simple model...")
        model = PySRRegressor(
            niterations=2,
            binary_operators=["+", "*"],
            unary_operators=["square"],
            population_size=10,
            populations=3,
            progress=False,
            verbosity=0,
            timeout_in_seconds=30
        )
        print("   ✅ Model created")
        
        print("   Generating test data...")
        X = np.array([[1], [2], [3], [4], [5]])
        y = X.ravel() ** 2
        print("   ✅ Test data created")
        
        print("   Training (2 iterations, 30s timeout)...")
        print("   This may take 15-30 seconds on first run...")
        
        try:
            model.fit(X, y)
            print("   ✅ Training completed!")
            
            # Try prediction
            pred = model.predict([[6]])
            print(f"   Test prediction: 6² = {pred[0]:.2f} (expected: 36)")
            
            if abs(pred[0] - 36) < 10:
                print("   ✅ Prediction reasonable")
            else:
                print("   ⚠️  Prediction seems off (normal for 2 iterations)")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Training failed: {e}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_mcp():
    """Check MCP packages"""
    print("\n📡 Checking MCP packages...")
    try:
        import fastmcp
        print(f"   FastMCP installed")
        print("   ✅ FastMCP OK")
    except ImportError:
        print("   ❌ FastMCP not installed")
        print("   Install with: pip install fastmcp")
        return False
    
    try:
        import mcp
        print(f"   MCP SDK installed")
        print("   ✅ MCP SDK OK")
        return True
    except ImportError:
        print("   ❌ MCP SDK not installed")
        print("   Install with: pip install mcp")
        return False


def install_pysr_backend():
    """Guide user through PySR backend installation"""
    print("\n🔧 PySR Backend Installation")
    print("=" * 60)
    print("PySR needs to install Julia packages on first use.")
    print("This is a ONE-TIME setup that takes 2-5 minutes.")
    print()
    
    response = input("Would you like to install PySR backend now? (y/n): ")
    
    if response.lower() == 'y':
        print("\n📦 Installing PySR backend...")
        print("This will download and compile Julia packages.")
        print("Please be patient, this only happens once!\n")
        
        try:
            import pysr
            pysr.install()
            print("\n✅ PySR backend installed successfully!")
            return True
        except Exception as e:
            print(f"\n❌ Installation failed: {e}")
            print("\nTry manual installation:")
            print('   python -c "import pysr; pysr.install()"')
            return False
    else:
        print("\n⚠️  Skipping backend installation")
        print('   Run later with: python -c "import pysr; pysr.install()"')
        return False


def main():
    """Run all checks"""
    print("=" * 60)
    print("PySR MCP Server - Diagnostic Tool")
    print("=" * 60)
    
    all_ok = True
    
    # Basic checks
    all_ok &= check_python()
    all_ok &= check_julia()
    all_ok &= check_numpy()
    all_ok &= check_pysr()
    all_ok &= check_mcp()
    
    print("\n" + "=" * 60)
    
    if not all_ok:
        print("⚠️  Some checks failed!")
        print("\nHowever, if PySR can find Julia, you may still be OK.")
        print("The most important check is the functionality test below.")
        print("\nIf you want to fix PATH issues:")
        print("   pip install -r requirements.txt")
        print("\nAnd ensure Julia is in your system PATH (optional).")
    else:
        print("✅ All basic checks passed!")
    
    print("=" * 60)
    
    # Offer to install backend
    print("\n" + "=" * 60)
    response = input("\nRun PySR functionality test? (y/n): ")
    
    if response.lower() == 'y':
        if test_pysr_basic():
            print("\n" + "=" * 60)
            print("🎉 Everything is working!")
            print("=" * 60)
            print("\nYou're ready to use the PySR MCP server!")
            print("\nNext steps:")
            print("   1. Run: python quick_start.py")
            print("   2. Run: python pysr_client.py")
        else:
            print("\n" + "=" * 60)
            print("⚠️  PySR test failed")
            print("=" * 60)
            print("\nPySR backend might not be installed.")
            install_pysr_backend()
    else:
        print("\nSkipping functionality test.")
        print("\n⚠️  Note: PySR backend must be installed before use!")
        print('   Run: python -c "import pysr; pysr.install()"')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()