import torch
import torch.nn as nn

_model = None

# model class required to load the weights
class _AestheticMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.2),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)

# same lazy loading as person_detect.py and embed.py
def _get_model(checkpoint_path):
    global _model
    if _model is None:
        _model = _AestheticMLP()
        _model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        _model.eval()
    return _model


def score_image(embedding, checkpoint_path):
    model = _get_model(checkpoint_path)
    tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        score = model(tensor)
    return score.item()
