#!/bin/bash

# Crawl2Answer - Start Script
# This script sets up the environment and starts the API server

echo "=== Crawl2Answer Startup Script ==="
echo "Crawl. Retrieve. Answer."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VENV_NAME="crawl2answer_env"
PYTHON_VERSION="python3"
API_HOST="0.0.0.0"
API_PORT="8000"

# Check if Python is installed
echo -e "${BLUE}Checking Python installation...${NC}"
if ! command -v $PYTHON_VERSION &> /dev/null; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3.8 or higher.${NC}"
    exit 1
fi

PYTHON_VER=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "${GREEN}Python $PYTHON_VER found${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_NAME" ]; then
    echo -e "${BLUE}Creating virtual environment...${NC}"
    $PYTHON_VERSION -m venv $VENV_NAME
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create virtual environment${NC}"
        exit 1
    fi
    echo -e "${GREEN}Virtual environment created${NC}"
else
    echo -e "${GREEN}Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source $VENV_NAME/bin/activate

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip

# Install requirements
echo -e "${BLUE}Installing requirements...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install requirements${NC}"
        exit 1
    fi
    echo -e "${GREEN}Requirements installed successfully${NC}"
else
    echo -e "${RED}requirements.txt not found${NC}"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}No .env file found. Creating from .env.example...${NC}"
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${YELLOW}Please edit .env file with your configuration${NC}"
    else
        echo -e "${RED}.env.example not found${NC}"
        exit 1
    fi
fi

# Create data directories
echo -e "${BLUE}Creating data directories...${NC}"
mkdir -p data/{raw,processed,embeddings}

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start the API server
echo -e "${GREEN}Starting Crawl2Answer API server...${NC}"
echo -e "${BLUE}API will be available at: http://localhost:$API_PORT${NC}"
echo -e "${BLUE}API Documentation: http://localhost:$API_PORT/docs${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Start with uvicorn
python -m uvicorn api.main:app --host $API_HOST --port $API_PORT --reload

# Deactivate virtual environment when done
deactivate