# Use the official Python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy files
RUN pip install requests
RUN pip install streamlit

# Copy the Streamlit app
COPY . .

# Expose port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
