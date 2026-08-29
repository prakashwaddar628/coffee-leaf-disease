import subprocess
import sys
import time
import os

def main():
    print("Starting Coffee Leaf Disease Detection System...")
    
    # Define paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(root_dir, "frontend")
    venv_python = os.path.join(root_dir, "venv", "Scripts", "python.exe")
    
    # Fallback to system python if venv python doesn't exist
    if not os.path.exists(venv_python):
        print(f"Warning: Virtual environment python not found at {venv_python}.")
        print("Falling back to system 'python'. Please ensure dependencies are installed.")
        venv_python = "python"
        
    try:
        # 1. Start FastAPI Backend
        print("\n=> Starting FastAPI Backend on http://localhost:8000...")
        backend_process = subprocess.Popen(
            [venv_python, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"],
            cwd=root_dir
        )
        
        # Give backend a moment to initialize
        time.sleep(2)
        
        # 2. Start Next.js Frontend
        print("\n=> Starting Next.js Frontend on http://localhost:3000...")
        
        # Use npm.cmd on Windows, npm on Unix
        npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
        
        frontend_process = subprocess.Popen(
            [npm_cmd, "run", "dev"],
            cwd=frontend_dir
        )
        
        print("\n" + "="*50)
        print("System is running!")
        print("- Frontend: http://localhost:3000")
        print("- Backend API: http://localhost:8000/docs")
        print("Press Ctrl+C to stop both servers.")
        print("="*50 + "\n")
        
        # Wait for processes to complete (they usually run indefinitely)
        backend_process.wait()
        frontend_process.wait()
        
    except KeyboardInterrupt:
        print("\nShutting down servers...")
    finally:
        # Ensure processes are terminated when script exits
        print("Stopping backend...")
        if 'backend_process' in locals() and backend_process.poll() is None:
            backend_process.terminate()
            
        print("Stopping frontend...")
        if 'frontend_process' in locals() and frontend_process.poll() is None:
            frontend_process.terminate()
            
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
