import ast
import importlib
from pathlib import Path


def main():
    required = ['torch', 'numpy', 'pandas', 'cv2', 'albumentations', 'scipy', 'PIL']
    for name in required:
        module = importlib.import_module(name)
        print(f'[OK] import {name}: {getattr(module, "__version__", "available")}')

    import torch
    from models.uspg_net import build_model
    from models.modules.dsp import DomainStylePerturbation

    x = torch.randn(2, 3, 128, 128)
    target = torch.randint(0, 2, (2, 1, 128, 128)).float()
    for model_type in ('baseline', 'dsp', 'spc', 'uspg'):
        torch.manual_seed(42)
        model = build_model(model_type)
        model.train()
        if model_type == 'baseline':
            output = model(x)
            assert output.shape == target.shape
        else:
            output = model(x, return_all=True)
            assert output['pred_orig'].shape == target.shape
            assert torch.isfinite(output['pred_orig']).all()
            if model_type == 'uspg':
                uncertainty = output['uncertainty_map']
                assert uncertainty.max() > 0, 'training uncertainty is identically zero'
        print(f'[OK] {model_type} forward')

    model = build_model('uspg')
    dropout_count = sum(isinstance(m, torch.nn.Dropout2d) for m in model.modules())
    assert dropout_count > 0
    print(f'[OK] MC-dropout layers: {dropout_count}')

    dsp = DomainStylePerturbation(prob=1.0).train()
    y = dsp(x)
    assert y.shape == x.shape and torch.isfinite(y).all() and not torch.equal(x, y)
    print('[OK] DSP produces finite, changed inputs')
    print('All corrected pipeline smoke tests passed.')


if __name__ == '__main__':
    main()
