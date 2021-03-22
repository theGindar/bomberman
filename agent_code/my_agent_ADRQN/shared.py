import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f'current device: {torch.cuda.get_device_name(torch.cuda.current_device())}')

EPS_START = 0.4
EPS_END = 0.1
EPS_DECAY = 10
