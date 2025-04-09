# Choose a specific Python base image (slim versions are smaller)
# Using Python 3.10 as an example, adjust if needed.
FROM python:3.10-slim

LABEL maintainer="Your Name <you@example.com>"
LABEL description="Docker image for pyATS MCP Server interacting via stdio"

# Install system dependencies required by pyATS, SSH, and your script
# Combine update, install, and cleanup in one layer to optimize image size
RUN echo "==> Installing System Dependencies (Python3, pip, SSH client, dos2unix)..." \
    && apt-get update \
    # Use --no-install-recommends to avoid installing unnecessary packages
    && apt-get install --no-install-recommends -y \
        # python3 and python3-pip might already be present/correct version in python image,
        # but explicitly listing ensures they are considered if needed by apt for other packages.
        # However, installing pip via apt here can sometimes conflict with the Python image's pip.
        # It's often safer to rely on the pip provided by the Python base image.
        # Let's comment out python3 and python3-pip install via apt for now.
        # python3 \
        # python3-pip \
        openssh-client \
        dos2unix \
        # Add build-essential in case pyats[full] needs to compile C extensions
        build-essential \
    # Clean up apt cache to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Upgrade pip and install Python dependencies
# Using --break-system-packages is necessary on newer Debian/Ubuntu base images with recent pip versions
# Using --no-cache-dir reduces image size by not storing the pip download cache
RUN echo "==> Upgrading pip and Installing pyATS[full] and other Python packages..." \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --break-system-packages \
        # Add pydantic and python-dotenv explicitly as they are direct dependencies of your script
        pydantic \
        python-dotenv \
        # Install pyats[full] - this can take a while and make the image large
        pyats[full]==25.2.0


# Copy your application code into the container's working directory
# This assumes pyats_mcp_server.py is in the same directory as the Dockerfile
COPY pyats_mcp_server.py .

# Optional: If you have other files needed by the script (e.g., commands.json), copy them too
# COPY commands.json .

# Expose port (Optional - Not needed for stdio interaction, but good practice if you switch to HTTP later)
# EXPOSE 8000

# Define the entrypoint to run your server script
# This will execute `python pyats_mcp_server.py` when the container starts
# It will run in continuous stdio mode by default because we removed the --oneshot logic as default
ENTRYPOINT ["python", "pyats_mcp_server.py"]

# CMD can provide default arguments to the ENTRYPOINT if needed.
# For example, if you wanted the default to be --oneshot (not recommended for the server):
# CMD ["--oneshot"]
# For the default continuous mode, no CMD arguments are needed:
CMD []