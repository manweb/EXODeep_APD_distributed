export IMAGE_FAMILY="tf-latest-cu92"
export ZONE="us-central1-b"
export INSTANCE_NAME="manuel-master-0 manuel-worker-0 manuel-worker-1 manuel-ps-0"
export INSTANCE_TYPE="n1-standard-8"
gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --machine-type=$INSTANCE_TYPE \
        --boot-disk-size=120GB \
