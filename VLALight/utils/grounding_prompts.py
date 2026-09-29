"""
VLM Grounding Prompt 模板

用于交通场景的车辆检测和车道级别计数
"""

# Implementation note.
GROUNDING_PROMPT_BASIC = """Detect all vehicles in this traffic intersection image.

IMPORTANT RULES:
- Only detect vehicles on the LEFT side of the yellow boundary line (approaching vehicles).
- Do NOT detect vehicles on the RIGHT side (departing vehicles).
- Each vehicle should have its own bounding box.

For each vehicle, output:
- "bbox_2d": [x1, y1, x2, y2] in relative coordinates (0-1000 scale)
- "label": vehicle type (e.g., "car", "truck", "bus")

Output ONLY a JSON array, no other text. Example format:
```json
[
    {"bbox_2d": [100, 200, 200, 350], "label": "car"},
    {"bbox_2d": [300, 250, 400, 400], "label": "truck"}
]
```"""


# Implementation note.
GROUNDING_PROMPT_ENHANCED = """Detect all vehicles in this traffic intersection image with spatial information.

DETECTION RULES:
- Only detect vehicles on the LEFT side of the yellow boundary line (approaching the intersection).
- Do NOT detect vehicles on the RIGHT side (departing from the intersection).
- Each vehicle should have its own bounding box.

OUTPUT FORMAT:
For each vehicle, provide:
- "bbox_2d": [x1, y1, x2, y2] in relative coordinates (0-1000 scale)
  - x1, y1: top-left corner
  - x2, y2: bottom-right corner
- "label": vehicle type ("car", "truck", "bus", etc.)
- "distance": distance from intersection ("near", "medium", "far") [OPTIONAL]

Output as a JSON array ONLY. Example:
```json
[
    {"bbox_2d": [365, 458, 440, 640], "label": "car", "distance": "near"},
    {"bbox_2d": [450, 458, 515, 630], "label": "car", "distance": "near"},
    {"bbox_2d": [390, 270, 455, 410], "label": "car", "distance": "medium"}
]
```"""


# Implementation note.
def get_direction_specific_prompt(direction: str) -> str:
    """
    获取方向特定的 Grounding Prompt
    
    Args:
        direction: 方向名称 (North, South, East, West)
        
    Returns:
        Grounding prompt 字符串
    """
    direction_hints = {
        'North': 'vehicles driving from the north side towards the intersection',
        'South': 'vehicles driving from the south side towards the intersection',
        'East': 'vehicles driving from the east side towards the intersection',
        'West': 'vehicles driving from the west side towards the intersection',
    }
    
    hint = direction_hints.get(direction, 'approaching vehicles')
    
    return f"""Detect all {hint}.

DETECTION RULES:
- Focus on vehicles on the LEFT side of the yellow boundary line (approaching).
- Ignore vehicles on the RIGHT side (departing).
- Detect each vehicle separately with its bounding box.

OUTPUT FORMAT (JSON array):
[
    {{"bbox_2d": [x1, y1, x2, y2], "label": "car/truck/bus"}},
    ...
]

Coordinates are relative (0-1000 scale). Output ONLY the JSON array."""


# Implementation note.
def build_grounding_prompt(
    direction: str = "Unknown",
    include_spatial: bool = False,
    custom_instructions: str = ""
) -> str:
    """
    构建完整的 Grounding Prompt
    
    Args:
        direction: 方向名称
        include_spatial: 是否包含空间信息（distance 字段）
        custom_instructions: 自定义指令
        
    Returns:
        完整的 prompt 字符串
    """
    if include_spatial:
        base_prompt = GROUNDING_PROMPT_ENHANCED
    else:
        base_prompt = GROUNDING_PROMPT_BASIC
    
    # Implementation note.
    if direction != "Unknown":
        direction_hint = f"\nDirection: This is the {direction} view of the intersection.\n"
        base_prompt = direction_hint + base_prompt
    
    # Implementation note.
    if custom_instructions:
        base_prompt = base_prompt + "\n\nADDITIONAL INSTRUCTIONS:\n" + custom_instructions
    
    return base_prompt


# Implementation note.
GROUNDING_PROMPT_TEST = """Test prompt: Detect all vehicles and output as JSON with bbox_2d and label."""
