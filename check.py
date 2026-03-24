import torch
state = torch.load(r'E:\kishan-sahyate-app\best_resnet.pth', map_location='cpu', weights_only=False)
keys = list(state.keys())
print("Total keys:", len(keys))
print("First 5:", keys[:5])
print("Last 5:", keys[-5:])