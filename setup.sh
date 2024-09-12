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
    print_colored $GREEN "Successfully connected to Kubernetes cluster."
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
    
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo update
    
    if ! helm install redis bitnami/redis \
        --namespace openfaas \
        --set auth.enabled=false \
        --set master.persistence.enabled=true \
        --set master.persistence.size=1Gi; then
        print_colored $RED "Failed to install Redis."
        exit 1
    fi
    
    print_colored $GREEN "Redis installed successfully."
}

deploy_functions() {
    print_colored $YELLOW "Deploying functions..."
    
    if [ ! -f stack.yml ]; then
        print_colored $RED "stack.yml not found. Please ensure it exists in the current directory."
        exit 1
    fi
    
    if ! faas-cli deploy -f stack.yml; then
        print_colored $RED "Failed to deploy functions."
        exit 1
    fi
    
    print_colored $GREEN "Functions deployed successfully."
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
    print_colored $YELLOW "Starting setup..."
    
    check_requirements
    check_kubernetes_connection
    
    install_openfaas
    wait_for_openfaas
    install_redis
    deploy_functions
    
    print_colored $GREEN "Setup completed successfully!"
    print_colored $YELLOW "Please ensure you update the 'stack.yml' file with your specific configurations if needed."
}

main