#!/bin/bash

set -eo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_colored() {
    color=$1
    message=$2
    echo -e "${color}${message}${NC}"
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

check_requirements() {
    print_colored $BLUE "Checking requirements..."
    local missing_tools=()

    for tool in kubectl helm faas-cli; do
        if ! command_exists $tool; then
            missing_tools+=("$tool")
        fi
    done

    if ! command_exists arkade; then
        print_colored $YELLOW "arkade not found. It will be installed during setup."
    fi

    if [ ${#missing_tools[@]} -ne 0 ]; then
        print_colored $RED "The following tools are required but not installed:"
        printf '  - %s\n' "${missing_tools[@]}"
        print_colored $RED "Please install these tools and try again."
        exit 1
    fi

    print_colored $GREEN "All required tools are installed."
}

check_kubernetes_connection() {
    print_colored $BLUE "Checking Kubernetes connection..."
    if ! kubectl cluster-info &>/dev/null; then
        print_colored $RED "Not connected to a Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    
    k8s_version=$(kubectl version --client --short)
    print_colored $GREEN "Connected to Kubernetes cluster. Client version: $k8s_version"
    
    print_colored $BLUE "Checking cluster resources..."
    nodes=$(kubectl get nodes --no-headers | wc -l)
    cpu=$(kubectl get nodes -o jsonpath='{.items[*].status.allocatable.cpu}' | tr ' ' '+' | bc)
    memory=$(kubectl get nodes -o jsonpath='{.items[*].status.allocatable.memory}' | sed 's/Ki/\/1024\/1024/g' | tr ' ' '+' | bc)
    
    print_colored $GREEN "Cluster resources:"
    echo "  Nodes: $nodes"
    echo "  Total Allocatable CPU: $cpu cores"
    echo "  Total Allocatable Memory: $memory GB"
}

install_arkade() {
    if ! command_exists arkade; then
        print_colored $YELLOW "Installing arkade..."
        if curl -sLS https://get.arkade.dev | sudo sh; then
            print_colored $GREEN "arkade installed successfully."
        else
            print_colored $RED "Failed to install arkade. Please install it manually."
            exit 1
        fi
    else
        print_colored $GREEN "arkade is already installed."
    fi
}

install_openfaas() {
    print_colored $BLUE "Installing OpenFaaS..."
    
    for namespace in openfaas openfaas-fn; do
        if ! kubectl get namespace $namespace &>/dev/null; then
            kubectl create namespace $namespace
        fi
    done
    
    print_colored $YELLOW "Deploying OpenFaaS..."
    if ! arkade install openfaas; then
        print_colored $RED "Failed to deploy OpenFaaS."
        exit 1
    fi
    
    print_colored $GREEN "OpenFaaS installed successfully."
}

install_redis() {
    print_colored $BLUE "Installing Redis..."
    
    local redis_values="redis/redis-values.yaml"
    if [ ! -f "$redis_values" ]; then
        print_colored $RED "$redis_values not found. Please ensure it exists in the project directory."
        exit 1
    fi
    
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo update
    
    if ! helm install redis bitnami/redis \
        --namespace openfaas \
        -f "$redis_values"; then
        print_colored $RED "Failed to install Redis."
        exit 1
    fi
    
    print_colored $GREEN "Redis installed successfully."
}

wait_for_openfaas() {
    print_colored $BLUE "Waiting for OpenFaaS to be ready..."
    local timeout=300
    
    if kubectl wait --for=condition=available --timeout=${timeout}s deployment/gateway -n openfaas; then
        print_colored $GREEN "OpenFaaS is ready."
    else
        print_colored $RED "Timeout waiting for OpenFaaS to be ready."
        exit 1
    fi
}

display_openfaas_info() {
    print_colored $BLUE "OpenFaaS Information:"
    
    local password=$(kubectl get secret -n openfaas basic-auth -o jsonpath="{.data.basic-auth-password}" | base64 --decode)
    echo "Username: admin"
    echo "Password: $password"
    
    local gateway_url=$(kubectl get svc -n openfaas gateway-external -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ -z "$gateway_url" ]; then
        gateway_url=$(kubectl get svc -n openfaas gateway-external -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    fi
    
    if [ -n "$gateway_url" ]; then
        echo "Gateway URL: http://$gateway_url:8080"
    else
        print_colored $YELLOW "Could not determine OpenFaaS gateway URL. Please check your service configuration."
    fi
}

main() {
    print_colored $YELLOW "Starting infrastructure setup..."
    
    check_requirements
    check_kubernetes_connection
    install_arkade
    install_openfaas
    wait_for_openfaas
    install_redis
    display_openfaas_info
    
    print_colored $GREEN "Infrastructure setup completed successfully!"
    print_colored $YELLOW "You can now run deploy_functions.sh to deploy your functions."
}

main