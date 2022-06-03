# deploy GKE marketplace application
# storage bucket
# gs://iunu_model_repo_for_triton_gke/iunu_model_repository
# https://console.cloud.google.com/marketplace/details/nvidia-ngc-public/triton-inference-server

# get the ingress host and port
export INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')

locust -f locustfile_convnext_onnx.py -H http://${INGRESS_HOST}:${INGRESS_PORT}