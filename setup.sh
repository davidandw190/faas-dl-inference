#!/bin/bash

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color :)

print_colored() {
    color=$1
    message=$2
    echo -e "${color}${message}${NC}"
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

check_requirements() {
    print_colored $YELLOW "Checking requirements..."
    local missing_tools=()

    if ! command_exists kubectl; then
        missing_tools+=("kubectl")
    fi
    if ! command_exists helm; then
        missing_tools+=("helm")
    fi
    if ! command_exists faas-cli; then
        missing_tools+=("faas-cli")
    fi
    if ! command_exists arkade; then
        print_colored $YELLOW "arkade not found. It will be installed during setup."
    fi

    if [ ${#missing_tools[@]} -ne 0 ]; then
        print_colored $RED "The following tools are required but not installed:"
        for tool in "${missing_tools[@]}"; do
            echo "- $tool"
        done
        print_colored $RED "Please install these tools and try again."
        exit 1
    fi

    print_colored $GREEN "All required tools are installed."
}

check_kubernetes_connection() {
    print_colored $YELLOW "Checking Kubernetes connection..."
    if ! kubectl cluster-info &>/dev/null; then
        print_colored $RED "Not connected to a Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    
    k8s_version=$(kubectl version --short | grep "Server Version" | cut -d " " -f3)
    print_colored $GREEN "Connected to Kubernetes cluster. Server version: $k8s_version"
    
    print_colored $YELLOW "Checking cluster resources..."
    nodes=$(kubectl get nodes -o name | wc -l)
    cpu=$(kubectl get nodes -o jsonpath='{.items[*].status.capacity.cpu}' | tr ' ' '+' | bc)
    memory=$(kubectl get nodes -o jsonpath='{.items[*].status.capacity.memory}' | sed 's/Ki/\/1024\/1024/g' | tr ' ' '+' | bc)
    
    print_colored $GREEN "Cluster resources:"
    echo "  Nodes: $nodes"
    echo "  Total CPU: $cpu cores"
    echo "  Total Memory: $memory GB"
}

install_openfaas() {
    print_colored $YELLOW "Installing OpenFaaS..."
    
    if ! command_exists arkade; then
        print_colored $YELLOW "Installing arkade..."
        curl -sLS https://get.arkade.dev | sudo sh
        print_colored $GREEN "arkade installed successfully."
    else
        print_colored $GREEN "arkade is already installed."
    fi
    
    if ! kubectl get namespace openfaas &>/dev/null; then
        kubectl create namespace openfaas
    fi
    if ! kubectl get namespace openfaas-fn &>/dev/null; then
        kubectl create namespace openfaas-fn
    fi
    
    print_colored $YELLOW "Deploying OpenFaaS..."
    if ! arkade install openfaas; then
        print_colored $RED "Failed to deploy OpenFaaS."
        exit 1
    fi
    
    print_colored $GREEN "OpenFaaS installed successfully."
}

install_redis() {
    print_colored $YELLOW "Installing Redis..."
    
    if [ ! -f "redis/redis-values.yaml" ]; then
        print_colored $RED "redis/redis-values.yaml not found. Please ensure it exists in the project directory."
        exit 1
    fi
    
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo update
    
    if ! helm install redis bitnami/redis \
        --namespace openfaas \
        -f redis/redis-values.yaml; then
        print_colored $RED "Failed to install Redis."
        exit 1
    fi
    
    print_colored $GREEN "Redis installed successfully."
}

wait_for_openfaas() {
    print_colored $YELLOW "Waiting for OpenFaaS to be ready..."
    local retries=0
    local max_retries=30
    
    while ! kubectl get pods -n openfaas | grep gateway | grep Running &>/dev/null; do
        if [ $retries -eq $max_retries ]; then
            print_colored $RED "Timeout waiting for OpenFaaS to be ready."
            exit 1
        fi
        sleep 10
        ((retries++))
        print_colored $YELLOW "Still waiting for OpenFaaS... (Attempt $retries/$max_retries)"
    done
    
    print_colored $GREEN "OpenFaaS is ready."
}

main() {
    print_colored $YELLOW "Starting infrastructure setup..."
    
    check_requirements
    check_kubernetes_connection
    
    install_openfaas
    wait_for_openfaas
    install_redis
    
    print_colored $GREEN "Infrastructure setup completed successfully!"
    print_colored $YELLOW "You can now run deploy_functions.sh to deploy your functions."
}

main