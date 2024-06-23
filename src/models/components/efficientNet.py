import torch
from torch import nn
import torchvision.models as models


class EfficientNet(nn.Module):

    def __init__(
        self,
        fc_output_size: int = 4,
    ) -> None:
        
        super().__init__()
        self.model = models.efficientnet_b4(pretrained = False) # shape ([1, 2048, 7, 7])
        self.fc_output_size = fc_output_size
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(1792, fc_output_size),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        batch_size, channels, width, height = x.size()
        x = self.model(x)

        return x


if __name__ == "__main__":
    _ = EfficientNet()
