import os
import sys
import subprocess
import logging
import requests
import time
from pyngrok import ngrok, conf
import uvicorn
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('local_deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure ngrok settings
def configure_ngrok():
    # Set a longer timeout for downloads (5 minutes)
    conf.get_default().request_timeout = 300.0
    
    # Disable update checks to prevent the warning
    os.environ["NGROK_NO_UPDATE_CHECK"] = "true"
    
    # Set the authtoken from environment variable
    ngrok_token = os.environ.get("NGROK_AUTHTOKEN")
    if ngrok_token:
        ngrok.set_auth_token(ngrok_token)
    else:
        logger.warning("NGROK_AUTHTOKEN not found in environment variables. Authentication may fail.")
    
    # Configure ngrok to use a different download URL if needed
    # Uncomment the following line if you want to use a different mirror
    # conf.get_default().ngrok_path = "path/to/your/downloaded/ngrok.exe"
    
    logger.info("Configured ngrok with extended timeout and disabled update checks")

async def main():
    # Check if .env file exists
    if not os.path.exists(".env"):
        logger.error(".env file not found. Please create it with the required API keys.")
        return
    
    # Configure ngrok
    configure_ngrok()
    
    # Set up ngrok tunnel
    try:
        port = int(os.environ.get("PORT", 8001))
        logger.info(f"Starting ngrok tunnel for port {port}...")
        
        # Try to connect with retry logic
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Specify http protocol instead of default https
                # Change this line in your main() function
                public_url = ngrok.connect(port, "http", bind_tls=True).public_url
                logger.info(f"ngrok tunnel established at: {public_url}")
                
                # Set environment variable for app.py to use
                os.environ["PUBLIC_URL"] = public_url
                break
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise e
                
                logger.warning(f"ngrok connection attempt {retry_count} failed: {e}")
                logger.info(f"Retrying in 5 seconds...")
                await asyncio.sleep(5)
        
        # Start the FastAPI application
        config = uvicorn.Config("app:app", host="0.0.0.0", port=port, reload=True)
        server = uvicorn.Server(config)
        await server.serve()
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        
        # Provide helpful instructions for manual ngrok setup
        if "download ngrok" in str(e).lower():
            logger.info("\nManual ngrok setup instructions:")
            logger.info("1. Download ngrok from https://ngrok.com/download")
            logger.info("2. Extract the ngrok.exe file")
            logger.info("3. Place it in a directory in your PATH or specify its location in this script")
            logger.info("4. Run this script again")
    finally:
        # Clean up ngrok tunnel
        ngrok.kill()

if __name__ == "__main__":
    # Run the async main function
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())