# Use an official Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy the entire project (excluding .venv)
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Jupyter if needed
RUN pip install jupyter

# Expose Jupyter Notebook port (optional)
EXPOSE 8888

# Set the command to run your notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]

ENV PYTHONPATH=/app
