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

deploy_functions() {
    print_colored $YELLOW "Deploying functions..."
    
    if [ ! -f stack.yml ]; then
        print_colored $RED "stack.yml not found. Please ensure it exists in the current directory."
        exit 1
    fi
    
    functions=$(grep "^  [a-zA-Z0-9-]\+:$" stack.yml | sed 's/://')
    
    if ! faas-cli deploy -f stack.yml; then
        print_colored $RED "Failed to deploy functions."
        exit 1
    fi
    
    print_colored $GREEN "All functions deployed successfully:"
    for func in $functions; do
        echo "  - $func"
    done
}

check_deployments() {
    print_colored $YELLOW "Checking deployments..."
    
    openfaas_pods=$(kubectl get pods -n openfaas -o name)
    print_colored $GREEN "OpenFaaS components deployed:"
    echo "$openfaas_pods" | sed 's/^/  /'
    
    redis_pods=$(kubectl get pods -n openfaas -l app=redis -o name)
    if [ -n "$redis_pods" ]; then
        print_colored $GREEN "Redis deployed:"
        echo "$redis_pods" | sed 's/^/  /'
    else
        print_colored $YELLOW "Redis not found in the openfaas namespace."
    fi
    
    print_colored $GREEN "Deployed functions:"
    kubectl get functions -n openfaas-fn -o custom-columns=NAME:.metadata.name,REPLICAS:.spec.replicas,AVAILABLE:.status.availableReplicas
}

main() {
    print_colored $YELLOW "Starting function deployment..."
    
    deploy_functions
    check_deploymentss
    
    print_colored $GREEN "Function deployment completed successfully!"
}

main