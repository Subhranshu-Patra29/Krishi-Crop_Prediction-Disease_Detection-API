import torch
from model_def import ResNet9

# Load model
model = ResNet9(3, 38)
model.load_state_dict(torch.load('res/plant-disease-model.pth', map_location='cpu'))
model.eval()

# Convert to TorchScript
example_input = torch.randn(1, 3, 256, 256)
traced_model = torch.jit.trace(model, example_input)

# Save TorchScript model
traced_model.save("res/plant_disease_model_mobile.pt")
