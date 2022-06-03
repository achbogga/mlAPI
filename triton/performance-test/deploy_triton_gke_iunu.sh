#!/usr/bin/env bash
conda activate triton_clients
export PROJECT_ID="production-233800"
export ZONE="us-central1-a"
export REGION="us-central1"
export DEPLOYMENT_NAME="triton-gke"

gcloud beta container clusters create ${DEPLOYMENT_NAME} \
--addons=HorizontalPodAutoscaling,HttpLoadBalancing,Istio \
--machine-type=n1-standard-8 --node-locations=${ZONE} \
--zone=${ZONE} --subnetwork=default --scopes cloud-platform \
--num-nodes 1 --project ${PROJECT_ID}

gcloud container node-pools create accel   \
--project ${PROJECT_ID}   \
--zone ${ZONE}   \
--cluster ${DEPLOYMENT_NAME}   --num-nodes 2   \
--accelerator type=nvidia-tesla-t4,count=1   \
--enable-autoscaling --min-nodes 2 --max-nodes 4   \
--machine-type n1-standard-4   --disk-size=100   \
--scopes cloud-platform   --verbosity error

# so that you can run kubectl locally to the cluster
gcloud container clusters get-credentials ${DEPLOYMENT_NAME} \
--project ${PROJECT_ID} --zone ${ZONE}

# deploy NVIDIA device plugin for GKE to prepare GPU nodes for driver install
kubectl apply -f \
https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
# make sure you can run kubectl locally to access the cluster
kubectl create clusterrolebinding cluster-admin-binding \
--clusterrole cluster-admin --user "$(gcloud config get-value account)"
# enable stackdriver custom metrics adaptor
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/k8s-stackdriver/master/custom-metrics-stackdriver-adapter/deploy/production/adapter.yaml
