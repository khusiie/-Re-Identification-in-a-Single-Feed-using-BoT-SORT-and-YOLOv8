import torchreid
import torch

# Load the pretrained model
model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=1000,
    pretrained=True
)

# Set the model to evaluation mode
model.eval()

# Save the model weights
torch.save(model.state_dict(), 'osnet_x1_0_market1501.pt')

print("✅ Saved as osnet_x1_0_market1501.pt")
