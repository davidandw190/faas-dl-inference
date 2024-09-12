# [WIP] FaaS Deep Learning Infernece

This repository represents part of my work towards better understanding how the FaaS paradigm can be adapted in Edge-Cloud environments for improving distributed, containerized ML inferencing workflows

## Features

- Functions are configured using OpenFaaS over Kubernetes, designed to be atomic, stateless, and respect FaaS standards.
- Atomic functions enable either standalone invocations or invocations within a workflow, coordinated by an orchestrator function.
- Function constraints (e.g., exec_timeout, write_timeout) are set considering edge environment limitations.
- ML models are in ONNX, quantized int8-ONNX, or Caffe format. Some models were directly used in ONNX format, while others were fine-tuned and converted.
- Lightweight and multi-stage Docker images are used, but improvemetns are still being made.
- Redis is used for caching of inference responses.

## Functions

- **sentiment-analysis** - uses a custom fine-tuned DistilBERT model to classify text into six emotions and returns the top three predicted emotions with their probabilities. It expects a JSON input with a "text" field, and uses an ONNX converted and int-8 qunatized model.

- **multi-label-sentiment-analysis** - uses a custom fine-tuned DistilBERT model, this time  with 28 emotion categories, and then returns the top three predicted emotions with their probabilities. It again expects a JSON input with a "text" field, and uses an ONNX converted and int-8 qunatized model.

- **text-classification** - uses a custom fine-tuned DistilBERT model on the AG News dataset. It expects JSON input with a "text" field and returns the predicted category, category ID, probability, and probabilities for all categories(World, Sports, Business, and Sci/Tech). The function uses an ONNX converted and int-8 qunatized model.

- **face-detection** - it expects a JSON input with an "image" field containing a hex-encoded image, and uses a pre-trained and quantized ONNX model to detect faces, and returns the number of faces detected, along with each face's bounding box, confidence score, and cropped face image (hex-encoded).

- **face-gender-detection** - it expects JSON input containing a list of faces, each with a face image (hex-encoded). The function uses a fine-tuned coffemodel GoogLeNet model to predict gender for each face, returning the number of faces processed and gender results (predicted gender and confidence) for each face.

- **face-emotion-detection** - it also expects JSON input containing a list of faces, each with a face image (hex-encoded) and uses the `emotion-ferplus-8` ONNX model to predict emotions for each face. It returns the number of faces processed and emotion results (predicted emotion, confidence, and probabilities for all emotions) for each face.

- **face-analysis-orchestrator** - it orchestrates a face analysis workflow, combining `face detection`, `gender detection`, and `emotion detection`. It processes input images, caches results of the infernce using Redis, and asynchronously calls other serverless functions for each analysis step. The workflow includes image preprocessing, face detection, a parallel gender and emotion detection done on the identified, cropped faces from the face detection, and result combination. 


## Setup

1. Clone the repository:

``` bash 
git clone https://github.com/davidandw190/faas-dl-inference.git && cd faas-dl-inference
```

2. Ensure you have the required infrastructure up and running:

``` bash
chmod +x setup.sh && ./setup.sh
```

3. From this point, you can choose to deploy the functions you want:
    *  You can deploy all the functions available by running:
    ```bash
    chmod +x setup.sh && ./deploy_all_functions.sh
    ```

    * You can only deploy the `face-anlaysis-workfload` and corresponding functions:
    ```bash
    chmod +x setup.sh && ./deploy_face_analysis_workflow.sh
    ```
    * Or you can independently deploy functions via:
    ```bash
    faas-cli deploy -f stack.yml --filter <desired-function> --skip-push --skip-build
    ```

4. Now you can invoke the desired deployed function:

```bash
curl -X POST http://<GATEWAY_URL>:8080/function/sentiment-analysis \
    -H "Content-Type: application/json" \
    -d '{"text": "I am feeling very happy today!"}'
```


