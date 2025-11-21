"""
HunyuanImage-3.0 ComfyUI Custom Nodes
Professional custom nodes for running Tencent's HunyuanImage-3.0 model in ComfyUI

Author: Eric Hiss (GitHub: EricRollei)
Contact: [eric@historic.camera, eric@rollei.us]
License: Dual License (Non-Commercial and Commercial Use)
Copyright (c) 2025 Eric Hiss. All rights reserved.

Dual License:
1. Non-Commercial Use: This software is licensed under the terms of the
   Creative Commons Attribution-NonCommercial 4.0 International License.
   To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/
   
2. Commercial Use: For commercial use, a separate license is required.
   Please contact Eric Hiss at [eric@historic.camera, eric@rollei.us] for licensing options.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
PARTICULAR PURPOSE AND NONINFRINGEMENT.

Note: HunyuanImage-3.0 model is subject to Tencent's Apache 2.0 license.
"""

from .hunyuan_quantized_nodes import NODE_CLASS_MAPPINGS as QUANTIZED_MAPPINGS
from .hunyuan_quantized_nodes import NODE_DISPLAY_NAME_MAPPINGS as QUANTIZED_DISPLAY_MAPPINGS

from .hunyuan_full_bf16_nodes import NODE_CLASS_MAPPINGS as FULL_MAPPINGS
from .hunyuan_full_bf16_nodes import NODE_DISPLAY_NAME_MAPPINGS as FULL_DISPLAY_MAPPINGS

from .hunyuan_api_nodes import NODE_CLASS_MAPPINGS as API_MAPPINGS
from .hunyuan_api_nodes import NODE_DISPLAY_NAME_MAPPINGS as API_DISPLAY_MAPPINGS

# Combine all mappings
NODE_CLASS_MAPPINGS = {
    **QUANTIZED_MAPPINGS,
    **FULL_MAPPINGS,
    **API_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **QUANTIZED_DISPLAY_MAPPINGS,
    **FULL_DISPLAY_MAPPINGS,
    **API_DISPLAY_MAPPINGS,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']