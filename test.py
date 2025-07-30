from PIL import Image
from scripts.agents.function_parser import parse_function_call
import json
from scripts.agents.smolvlm2_collator import create_vlm_collate_fn

if __name__ == "__main__":
    from transformers import AutoProcessor
    from datasets import load_dataset, IterableDataset

    class ScriptArguments:
        image_resize = {
            "factor": 28,
            "min_pixels": 200704,
            "max_pixels": 1003520,
        }

    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct", model_revision="main")
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", model_revision="main", min_pixels=200704, max_pixels=1003520)
    collate_fn = create_vlm_collate_fn(processor, ScriptArguments())
    max = 0
    answer_token = 0
    user_token = 0
    system_token = 0
    total_token = 0
    for dataset_name in ['aitw', 'amex', 'android_control', 'coat', 'gui-odyssey', 'guiact-web-multi', 'guiact-web-single', 'mind2web', 'miniwob']:
        datasets = load_dataset("smolagents/aguvis-stage-2", dataset_name, split="train")
        for i, example in enumerate(datasets):
            batch = collate_fn([example])
            exit()
    
    # print(batch)