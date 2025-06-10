from transformers import ViTModel

from torch import nn


class ViT_model(nn.Module):
    def __init__(self, model_name, num_classes = 10):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)
        
    def forward(self, x):
        outputs = self.vit(**x)
        last_hidden_states = outputs.last_hidden_state

        x = self.classifier(last_hidden_states[:, 0])
        return x
