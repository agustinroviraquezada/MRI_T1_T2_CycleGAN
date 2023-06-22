# Use a base image with Python pre-installed
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the repository files to the working directory
COPY . /app

# Install virtualenv
RUN pip install --upgrade pip && \
    pip install virtualenv

# Create and activate a virtual environment
RUN virtualenv venv && \
    source venv/bin/activate

# Install dependencies inside the virtual environment
RUN pip install -r requirements.txt

# Set environment variables
ENV INPUT_FILE="" \
    MODEL_FILE="" \
    MODALITY="" \
    HDBET="" \
    FILTERS="" \
    DIM="" \
    CROP="" \
    OPERATION="" \
    BLACK_PROPORTION=""

# Run the script when the container starts
CMD python cycle/Test.py --input "$INPUT_FILE" --model "$MODEL_FILE" --Modality "$MODALITY" --HDBET "$HDBET" --filters "$FILTERS" --dim "$DIM" --crop "$CROP" --operation "$OPERATION" --BlackProportion "$BLACK_PROPORTION"
