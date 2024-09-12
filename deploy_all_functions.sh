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

check_prerequisites() {
    print_colored $BLUE "Checking prerequisites..."
    
    if ! command -v faas-cli &> /dev/null; then
        print_colored $RED "faas-cli could not be found. Please install OpenFaaS CLI."
        exit 1
    fi

    if ! command -v kubectl &> /dev/null; then
        print_colored $RED "kubectl could not be found. Please install kubectl."
        exit 1
    fi
    
    if [ ! -f stack.yml ]; then
        print_colored $RED "stack.yml not found. Please ensure it exists in the current directory."
        exit 1
    fi
}

get_functions() {
    grep "^  [a-zA-Z0-9-]\+:$" stack.yml | sed 's/://' | sort
}

deploy_functions() {
    print_colored $YELLOW "Deploying functions..."
    
    functions=$(get_functions)
    
    if ! faas-cli deploy -f stack.yml --skip-push --skip-build; then
        print_colored $RED "Failed to deploy functions."
        exit 1
    fi
    
    print_colored $GREEN "Functions deployed successfully:"
    echo "$functions" | sed 's/^/  - /'
}

check_deployments() {
    print_colored $YELLOW "Checking deployments..." 
    print_colored $GREEN "Deployed functions:"
    
    functions=$(get_functions)
    
    for func in $functions; do
        status=$(kubectl get functions -n openfaas-fn $func -o custom-columns=NAME:.metadata.name,REPLICAS:.spec.replicas,AVAILABLE:.status.availableReplicas --no-headers)
        if [ -z "$status" ]; then
            print_colored $RED "$func: Not deployed"
        else
            echo "$status"
        fi
    done
}

wait_for_functions_ready() {
    print_colored $YELLOW "Waiting for functions to be ready..."
    
    functions=$(get_functions)
    
    for func in $functions; do
        print_colored $BLUE "Waiting for $func to be ready..."
        kubectl wait --for=condition=available --timeout=300s deployment -n openfaas-fn $func
    done
}

display_function_urls() {
    print_colored $YELLOW "Function URLs:"
    
    gateway_url=$(kubectl get svc -n openfaas gateway-external -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ -z "$gateway_url" ]; then
        gateway_url=$(kubectl get svc -n openfaas gateway-external -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    fi
    
    if [ -z "$gateway_url" ]; then
        print_colored $RED "Could not determine OpenFaaS gateway URL."
    else
        functions=$(get_functions)
        for func in $functions; do
            echo "$func: http://$gateway_url:8080/function/$func"
        done
    fi
}

main() {
    print_colored $YELLOW "Starting function deployment..."
    
    check_prerequisites
    deploy_functions
    wait_for_functions_ready
    check_deployments
    display_function_urls
    
    print_colored $GREEN "Function deployment completed successfully!"
}

main