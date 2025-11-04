# Set PYTHONPATH to include project root
$env:PYTHONPATH = "C:\Users\User\Projects\FYP2;$env:PYTHONPATH"

# Change to project root
Set-Location "C:\Users\User\Projects\FYP2"

# Run flwr with the provided arguments
& flwr run federated_pneumonia_detection/src/control/federated_new_version @args
