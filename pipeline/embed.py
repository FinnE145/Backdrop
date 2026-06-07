import open_clip
import torch

## laxy load model, same as person_detect.py
_model = None
_preprocess = None


def _get_model(model_name, checkpoint_path):
    global _model, _preprocess
    if _model is None:
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=checkpoint_path
        )
        _model.eval()   # puts model in inference mode
    return _model, _preprocess


def embed_image(image, model_name, checkpoint_path):
    model, preprocess = _get_model(model_name, checkpoint_path)

    # center crops to 224x224, normalizes to model inputs, adds batch dimension (1,3,224,224)
    tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():   # no gradient graph, as were not training
        features = model.encode_image(tensor)   # (1, 768) tensor
        features = features / features.norm(dim=-1, keepdim=True)   # normalize to unit length for cosine similarity

    return features.squeeze(0).tolist()     # remove the batch dimension (768,)
