#Loading the MNIST MODEL
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = MNISTNet()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()
print("Model loaded and ready for inference!")


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # ensure 1 channel
    transforms.Resize((28, 28)),                  # resize if needed
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def predict_mnist_probabilities(image_path: str) -> str:
    try:
        # Load and transform image
        image = Image.open(image_path).convert("L")  # convert to grayscale
        image = transform(image).unsqueeze(0)  # add batch dimension

        # Get probabilities
        with torch.no_grad():
            output = model(image)
            probabilities = torch.softmax(output, dim=1).squeeze().tolist()

        # Create a string representation
        result = "\n".join([f"{i}: {prob:.4f}" for i, prob in enumerate(probabilities)])
        return result
    
    except Exception as e:
        return f"Error: {e}"