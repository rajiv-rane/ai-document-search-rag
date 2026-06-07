# Use an official, lightweight Python image
FROM python:3.10-slim

# Create a non-root user (Required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for PyMuPDF and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY --chown=user . /app

# Ensure the non-root user has write access to runtime directories
RUN mkdir -p /app/dataset /app/vector_store_data \
    && chown -R user:user /app

# Switch to the non-root user
USER user

# Expose Hugging Face's strictly required port
EXPOSE 7860

# Command to run the Streamlit application on port 7860
ENTRYPOINT ["streamlit", "run", "ui/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
