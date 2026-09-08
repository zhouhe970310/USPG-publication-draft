def build_comparison_model(model_type):
    import os
    from pathlib import Path

    import torch
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            'Comparison models require segmentation-models-pytorch. '
            'Install the corrected requirements.txt first.'
        ) from exc
    common = dict(encoder_name='resnet18', encoder_weights=None,
                  in_channels=3, classes=1)
    if model_type == 'unet':
        model = smp.Unet(**common)
    if model_type == 'unetpp':
        model = smp.UnetPlusPlus(**common)
    if model_type == 'deeplabv3p':
        model = smp.DeepLabV3Plus(**common)
    if model_type not in {'unet', 'unetpp', 'deeplabv3p'}:
        raise ValueError(f'Unknown comparison model: {model_type}')
    default = (Path(os.environ.get('TORCH_HOME', Path.home() / '.cache' / 'torch')) /
               'hub' / 'checkpoints' / 'resnet18-5c106cde.pth')
    weight_path = Path(os.environ.get('USPG_RESNET18_WEIGHTS', default))
    if not weight_path.is_file():
        raise FileNotFoundError(f'Local ImageNet weights not found: {weight_path}')
    state = torch.load(weight_path, map_location='cpu', weights_only=False)
    model.encoder.load_state_dict(state, strict=False)
    return model
