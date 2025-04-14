FROM python:3.11-slim-buster

WORKDIR /app

# Install dependencies
RUN pip install aiohttp  
RUN pip install python-dotenv 
RUN pip install pydantic

# Copy the application code
COPY . . 

# Run the FastMCP server
CMD ["python", "-u", "server.py"]