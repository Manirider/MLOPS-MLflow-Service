import os
import torch
import torch.onnx
import onnx
import logging
from pathlib import Path
from typing import Optional, Tuple
logger = logging.getLogger(__name__)
def convert_to_onnx(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...] = (1, 1, 28, 28),
    output_path: str = "model.onnx",
    opset_version: int = 12
) -> Optional[str]:
    try:
        model.eval()
        dummy_input = torch.randn(
            input_shape, device=next(model.parameters()).device)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info(
            f"Model successfully converted to ONNX and saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"ONNX conversion failed: {e}")
        return None
