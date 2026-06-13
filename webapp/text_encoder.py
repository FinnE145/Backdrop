import numpy as np
import open_clip
import torch

_model = None
_tokenizer = None


def _get_model(model_name: str, checkpoint_path: str):
    global _model, _tokenizer
    if _model is None:
        _model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=checkpoint_path
        )
        _model.eval()
        _tokenizer = open_clip.get_tokenizer(model_name)
    return _model, _tokenizer


def encode_text(query: str, model_name: str, checkpoint_path: str) -> np.ndarray:
    model, tokenizer = _get_model(model_name, checkpoint_path)
    tokens = tokenizer([query])
    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze(0).numpy().astype(np.float32)
