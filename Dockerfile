# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the entire contents of the current directory to /app
COPY . /app

# Change to the 'automate' directory
WORKDIR /app/automate

# Install Streamlit (if not included in requirements.txt)
RUN pip install streamlit

# Command to run your Streamlit application
CMD ["streamlit", "run", "main.py"]
