"""
Main Application Launcher for OMR Evaluation System
Provides options to run backend, frontend, or both
"""
import os
import sys
import subprocess
import threading
import time
import requests
from pathlib import Path

class OMRSystemLauncher:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.base_dir = Path(__file__).parent
        
    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        print("🔍 Checking dependencies...")
        
        required_packages = [
            'fastapi', 'uvicorn', 'streamlit', 'sqlalchemy', 
            'opencv-python', 'numpy', 'pandas', 'pillow',
            'requests', 'plotly'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"✅ {package}")
            except ImportError:
                print(f"❌ {package}")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
            print("Please install them using:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
        
        print("✅ All dependencies are installed!")
        return True
    
    def setup_directories(self):
        """Create necessary directories"""
        print("📁 Setting up directories...")
        
        directories = ['uploads', 'processed', 'exports', 'logs']
        
        for directory in directories:
            dir_path = self.base_dir / directory
            dir_path.mkdir(exist_ok=True)
            print(f"✅ {directory}/")
        
        print("✅ Directories setup complete!")
    
    def initialize_database(self):
        """Initialize the database"""
        print("🗄️  Initializing database...")
        
        try:
            from database import init_database
            init_database()
            print("✅ Database initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            return False
    
    def start_backend(self):
        """Start the FastAPI backend server"""
        print("🚀 Starting backend server...")
        
        try:
            # Start backend in a separate process
            cmd = [sys.executable, "main.py"]
            self.backend_process = subprocess.Popen(
                cmd,
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait a moment for server to start
            time.sleep(3)
            
            # Check if backend is running
            if self.check_backend_health():
                print("✅ Backend server started successfully!")
                print("🌐 Backend API: http://localhost:8000")
                print("📚 API Documentation: http://localhost:8000/docs")
                return True
            else:
                print("❌ Backend server failed to start properly")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
            return False
    
    def start_frontend(self):
        """Start the Streamlit frontend"""
        print("🎨 Starting frontend application...")
        
        try:
            # Start frontend in a separate process
            cmd = [sys.executable, "-m", "streamlit", "run", "frontend.py", "--server.port=8501"]
            self.frontend_process = subprocess.Popen(
                cmd,
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait a moment for server to start
            time.sleep(5)
            
            print("✅ Frontend application started successfully!")
            print("🌐 Frontend URL: http://localhost:8501")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start frontend: {e}")
            return False
    
    def check_backend_health(self):
        """Check if backend is healthy"""
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def stop_services(self):
        """Stop all running services"""
        print("\n🛑 Stopping services...")
        
        if self.backend_process:
            self.backend_process.terminate()
            self.backend_process.wait()
            print("✅ Backend stopped")
        
        if self.frontend_process:
            self.frontend_process.terminate()
            self.frontend_process.wait()
            print("✅ Frontend stopped")
    
    def run_backend_only(self):
        """Run only the backend server"""
        print("🔧 Starting OMR Evaluation System - Backend Only")
        print("=" * 50)
        
        if not self.check_dependencies():
            return False
        
        self.setup_directories()
        
        if not self.initialize_database():
            return False
        
        if self.start_backend():
            print("\n✅ Backend is running!")
            print("Press Ctrl+C to stop the server")
            
            try:
                # Keep the main thread alive
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop_services()
                print("\n👋 Backend stopped!")
            
            return True
        
        return False
    
    def run_frontend_only(self):
        """Run only the frontend application"""
        print("🎨 Starting OMR Evaluation System - Frontend Only")
        print("=" * 50)
        
        if not self.check_dependencies():
            return False
        
        # Check if backend is running
        if not self.check_backend_health():
            print("⚠️  Backend server is not running!")
            print("Please start the backend first or use 'full' mode")
            return False
        
        if self.start_frontend():
            print("\n✅ Frontend is running!")
            print("Press Ctrl+C to stop the application")
            
            try:
                # Keep the main thread alive
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop_services()
                print("\n👋 Frontend stopped!")
            
            return True
        
        return False
    
    def run_full_system(self):
        """Run both backend and frontend"""
        print("🚀 Starting OMR Evaluation System - Full System")
        print("=" * 50)
        
        if not self.check_dependencies():
            return False
        
        self.setup_directories()
        
        if not self.initialize_database():
            return False
        
        # Start backend first
        if not self.start_backend():
            return False
        
        # Start frontend
        if not self.start_frontend():
            self.stop_services()
            return False
        
        print("\n🎉 OMR Evaluation System is fully operational!")
        print("🌐 Frontend: http://localhost:8501")
        print("🌐 Backend API: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop all services")
        
        try:
            # Keep the main thread alive
            while True:
                time.sleep(1)
                
                # Check if processes are still running
                if self.backend_process and self.backend_process.poll() is not None:
                    print("❌ Backend process died unexpectedly")
                    break
                
                if self.frontend_process and self.frontend_process.poll() is not None:
                    print("❌ Frontend process died unexpectedly")
                    break
                    
        except KeyboardInterrupt:
            self.stop_services()
            print("\n👋 All services stopped!")
        
        return True
    
    def show_help(self):
        """Show help information"""
        print("🔧 OMR Evaluation System Launcher")
        print("=" * 40)
        print("Usage: python launcher.py [mode]")
        print("\nAvailable modes:")
        print("  backend    - Start only the backend API server")
        print("  frontend   - Start only the frontend application")
        print("  full       - Start both backend and frontend (default)")
        print("  help       - Show this help message")
        print("\nExamples:")
        print("  python launcher.py")
        print("  python launcher.py full")
        print("  python launcher.py backend")
        print("  python launcher.py frontend")

def main():
    """Main function"""
    launcher = OMRSystemLauncher()
    
    # Get mode from command line arguments
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    
    if mode == "backend":
        launcher.run_backend_only()
    elif mode == "frontend":
        launcher.run_frontend_only()
    elif mode == "full":
        launcher.run_full_system()
    elif mode == "help":
        launcher.show_help()
    else:
        print(f"❌ Unknown mode: {mode}")
        launcher.show_help()

if __name__ == "__main__":
    main()