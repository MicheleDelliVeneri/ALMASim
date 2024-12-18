# Define variables
APP_DIR := app
TEST_DIR := tests
DOCKER_IMAGE := almasim:latest

# Install dependencies
install:
    pip install -r requirements.txt

# Run tests
test:
    pytest $(TEST_DIR) --cov=$(APP_DIR)

# Run the application
run:
    python $(APP_DIR)/main.py

# Build Docker image
docker-build:
    docker build -t $(DOCKER_IMAGE) .

# Run Docker container
docker-run:
    docker run -p 8000:8000 $(DOCKER_IMAGE)

# Clean up
clean:
    find . -type d -name "__pycache__" -exec rm -r {} +
    find . -type f -name "*.pyc" -delete