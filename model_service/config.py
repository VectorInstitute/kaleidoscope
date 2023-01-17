import socket

# Address of the Lingua gateway service, in the format "host:port"
GATEWAY_HOST = "llm.cluster.local:3001"

# Address of this model service, in the format "host:port". 
# By default this will be the external-facing IP address and port 8888, but we want the option to customize this.
hostname = socket.gethostname()
ip_addr = socket.gethostbyname(hostname)
MODEL_HOST = f"{ip_addr}:8888"

# Scheduler script invoked by the job runner
JOB_SCHEDULER_BIN = "~/lingua/model_service/job_schedulers/slurm.sh"

