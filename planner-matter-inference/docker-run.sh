#!/bin/bash

# Helper script to build and run the GUI Agent Docker container

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== GUI Agent Docker Helper ===${NC}"

# Function to check if docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed${NC}"
        echo "Please install Docker first: https://docs.docker.com/get-docker/"
        exit 1
    fi
}

# Function to check if nvidia-docker is available
check_nvidia_docker() {
    if ! docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}Warning: nvidia-docker might not be properly configured${NC}"
        echo "GPU support may not work. Install nvidia-docker2 for GPU support."
    else
        echo -e "${GREEN}GPU support detected${NC}"
    fi
}

# Function to build the Docker image
build_image() {
    echo -e "${GREEN}Building Docker image...${NC}"
    docker compose build
    echo -e "${GREEN}Build complete!${NC}"
}

# Function to run the container
run_container() {
    echo -e "${GREEN}Starting container...${NC}"
    docker compose up -d
    echo -e "${GREEN}Container started!${NC}"
    echo ""
    echo "To enter the container, run:"
    echo -e "${YELLOW}  docker exec -it gui-agent-container bash${NC}"
}

# Function to enter the container
enter_container() {
    echo -e "${GREEN}Entering container...${NC}"
    docker exec -it gui-agent-container bash
}

# Function to stop the container
stop_container() {
    echo -e "${GREEN}Stopping container...${NC}"
    docker compose down
    echo -e "${GREEN}Container stopped!${NC}"
}

# Main menu
show_menu() {
    echo ""
    echo "What would you like to do?"
    echo "1) Build Docker image"
    echo "2) Start container"
    echo "3) Enter container (bash)"
    echo "4) Stop container"
    echo "5) Build and run"
    echo "6) Exit"
    echo ""
}

# Check prerequisites
check_docker

# If no arguments, show interactive menu
if [ $# -eq 0 ]; then
    while true; do
        show_menu
        read -p "Enter choice [1-6]: " choice
        case $choice in
            1)
                build_image
                ;;
            2)
                check_nvidia_docker
                run_container
                ;;
            3)
                enter_container
                ;;
            4)
                stop_container
                ;;
            5)
                build_image
                check_nvidia_docker
                run_container
                echo ""
                echo -e "${GREEN}Ready to use! Enter the container with:${NC}"
                echo -e "${YELLOW}  ./docker-run.sh shell${NC}"
                ;;
            6)
                echo "Goodbye!"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid option${NC}"
                ;;
        esac
    done
else
    # Handle command line arguments
    case "$1" in
        build)
            build_image
            ;;
        start)
            check_nvidia_docker
            run_container
            ;;
        shell|bash|exec)
            enter_container
            ;;
        stop)
            stop_container
            ;;
        rebuild)
            stop_container
            build_image
            check_nvidia_docker
            run_container
            ;;
        *)
            echo "Usage: $0 {build|start|shell|stop|rebuild}"
            echo ""
            echo "Commands:"
            echo "  build    - Build the Docker image"
            echo "  start    - Start the container"
            echo "  shell    - Enter the container (bash)"
            echo "  stop     - Stop the container"
            echo "  rebuild  - Stop, rebuild, and restart"
            echo ""
            echo "Or run without arguments for interactive menu"
            exit 1
            ;;
    esac
fi
