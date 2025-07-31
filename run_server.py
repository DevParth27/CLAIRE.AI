#!/usr/bin/env python3
import os
import subprocess
import sys

def main():
    # Get port from environment variable, default to 8000
    port = os.environ.get('PORT', '8000')
    
    print(f"Raw PORT environment variable: '{port}'")
    
    # Validate port is a valid integer
    try:
        port_int = int(port)
        if port_int < 1 or port_int > 65535:
            raise ValueError("Port must be between 1 and 65535")
    except ValueError as e:
        print(f"Error: Invalid port value '{port}': {e}")
        print("Using default port 8000")
        port_int = 8000
    
    # Build uvicorn command
    cmd = [
        'python', '-m', 'uvicorn',
        'app:app',  # Replace 'app:app' with your actual app module:app_instance
        '--host', '0.0.0.0',
        '--port', str(port_int),
        '--workers', '1'
    ]
    
    print(f"Starting server on port {port_int}")
    print(f"Command: {' '.join(cmd)}")
    
    # Execute uvicorn
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting server: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("Server stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()