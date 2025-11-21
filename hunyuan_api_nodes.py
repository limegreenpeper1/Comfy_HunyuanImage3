import logging
import requests
import re
import json
from .hunyuan_api_config import get_api_config

logger = logging.getLogger(__name__)

class HunyuanPromptRewriter:
    """
    Standalone node for rewriting prompts using DeepSeek or compatible APIs.
    Uses configuration from api_config.ini or environment variables.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A serene landscape"}),
                "rewrite_style": (["en_recaption", "en_think_recaption", "none"], {"default": "en_recaption"}),
            },
            "optional": {
                "manual_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("rewritten_prompt",)
    FUNCTION = "rewrite_prompt"
    CATEGORY = "HunyuanImage3/Prompting"

    def rewrite_prompt(self, prompt, rewrite_style, manual_seed=0):
        if rewrite_style == "none" or not prompt.strip():
            return (prompt,)

        config = get_api_config()
        api_key = config.get("api_key")
        
        if not api_key:
            logger.warning("No API key found in config or environment. Returning original prompt.")
            return (prompt,)

        logger.info(f"Rewriting prompt using LLM API (style: {rewrite_style})...")
        
        # Official HunyuanImage-3.0 system prompts
        system_prompts = {
            "en_recaption": """You are a world-class image generation prompt expert. Your task is to rewrite a user's simple description into a **structured, objective, and detail-rich** professional-level prompt.

The final output must be wrapped in `<recaption>` tags.

### **Universal Core Principles**

When rewriting the prompt (inside the `<recaption>` tags), you must adhere to the following principles:

1.  **Absolute Objectivity**: Describe only what is visually present. Avoid subjective words like "beautiful" or "sad". Convey aesthetic qualities through specific descriptions of color, light, shadow, and composition.
2.  **Physical and Logical Consistency**: All scene elements (e.g., gravity, light, shadows, reflections, spatial relationships, object proportions) must strictly adhere to real-world physics and common sense.
3.  **Structured Description**: Strictly follow a logical order: from general to specific, background to foreground, and primary to secondary elements.
4.  **Use Present Tense**: Describe the scene from an observer's perspective using the present tense.
5.  **Use Rich and Specific Descriptive Language**: Use precise adjectives to describe quantity, size, shape, color, and other attributes.

### **Final Output Requirements**

1.  **Output the Final Prompt Only**: Do not show any thought process or formatting.
2.  **Adhere to the Input**: Retain the core concepts and attributes from the user's input.
3.  **Style Reinforcement**: Mention the core style 3-5 times within the prompt.
4.  **Avoid Self-Reference**: Describe the image content directly.
5.  **The final output must be wrapped in `<recaption>xxxx</recaption>` tags.**""",
            
            "en_think_recaption": """You will act as a top-tier Text-to-Image AI. Your task is to deeply analyze the user's text input and transform it into a detailed, artistic image description.

Your workflow is divided into two phases:

1. **Thinking Phase (<think>)**: Break down the image elements:
   - Subject: Core character(s) or object(s), appearance, posture, expression
   - Composition: Camera angle, layout (close-up, long shot, etc.)
   - Environment/Background: Scene location, time, weather
   - Lighting: Type, direction, quality of light source
   - Color Palette: Main color tone and scheme
   - Quality/Style: Artistic style and technical details
   - Details: Minute elements that enhance realism

2. **Recaption Phase (<recaption>)**: Merge all details into a coherent, precise description.

**Key Principles:**
- Absolutely Objective: Describe only what is visually present
- Physical and Logical Consistency: Follow real-world physics
- Structured Description: Whole to part, background to foreground
- Use Present Tense: "A man stands," "light shines on..."
- Rich and Specific Language: Precise adjectives, no vague expressions

**Output Format:**
<think>Thinking process</think><recaption>Refined image description</recaption>"""
        }

        system_prompt = system_prompts.get(rewrite_style, system_prompts["en_recaption"])
        
        api_url = config.get("api_url", "https://api.deepseek.com/v1/chat/completions")
        model_name = config.get("model_name", "deepseek-chat")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        # Pass seed if provided and non-zero (if API supports it, usually harmless if not)
        if manual_seed > 0:
            data["seed"] = manual_seed

        try:
            response = requests.post(api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            content = result['choices'][0]['message']['content']
            
            # Extract content between <recaption> tags
            match = re.search(r'<recaption>(.*?)</recaption>', content, re.DOTALL)
            if match:
                rewritten = match.group(1).strip()
                logger.info("Prompt rewritten successfully")
                return (rewritten,)
            else:
                # Fallback if tags are missing but content exists
                logger.warning("No <recaption> tags found in response, using full content")
                return (content.strip(),)
                
        except Exception as e:
            logger.error(f"Prompt rewriting failed: {e}")
            return (prompt,)

NODE_CLASS_MAPPINGS = {
    "HunyuanPromptRewriter": HunyuanPromptRewriter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanPromptRewriter": "Hunyuan Prompt Rewriter (DeepSeek)"
}
