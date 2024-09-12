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

deploy_face_analysis_functions() {
    print_colored $YELLOW "Deploying face analysis workflow functions..."
    
    if [ ! -f stack.yml ]; then
        print_colored $RED "stack.yml not found. Please ensure it exists in the current directory."
        exit 1
    fi
    
    workflow_functions=(
        "face-analysis-orchestrator"
        "face-detection"
        "face-gender-detection"
        "face-emotion-detection"
    )
    
    for func in "${workflow_functions[@]}"; do
        print_colored $YELLOW "Deploying $func..."
        if ! faas-cli deploy -f stack.yml --filter $func; then
            print_colored $RED "Failed to deploy $func."
            exit 1
        fi
        print_colored $GREEN "$func deployed successfully."
    done
}

check_face_analysis_deployments() {
    print_colored $YELLOW "Checking face analysis workflow function deployments..."
    
    workflow_functions=(
        "face-analysis-orchestrator"
        "face-detection"
        "face-gender-detection"
        "face-emotion-detection"
    )
    
    print_colored $GREEN "Deployed face analysis workflow functions:"
    for func in "${workflow_functions[@]}"; do
        kubectl get functions -n openfaas-fn $func -o custom-columns=NAME:.metadata.name,REPLICAS:.spec.replicas,AVAILABLE:.status.availableReplicas
    done
}

main() {
    print_colored $YELLOW "Starting face analysis workflow deployment..."
    
    deploy_face_analysis_functions
    check_face_analysis_deployments
    
    print_colored $GREEN "Face analysis workflow deployment completed successfully!"
}

main