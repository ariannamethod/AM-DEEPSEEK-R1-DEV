#!/usr/bin/env python3
"""
Script to download, process, and upload the aguvis-stage2 dataset.
Downloads from huggingface.co/datasets/xlangai/aguvis-stage2 and uploads to smolagents/aguvis-stage-2
"""

import re
import gc
import sys
import json
import os
import shutil
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Generator, Callable, Literal
from tqdm import tqdm
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, login, snapshot_download
from collections import defaultdict
from PIL import Image
import tarfile
from prompts import OS_SYSTEM_PROMPT, MOBILE_SYSTEM_PROMPT
from models import ConversationDataList, ConversationData, ChatMessage, DataRow
from function_parser import parse_function_call
from action_conversion import action_conversion


api = HfApi()

# aguvis json file with mobile action space
MOBILE_FILE = [
    "android_control.json",
    "gui-odyssey-l1.json",
    "aitw-l3.json",
    "coat.jsonamex-l2.json",
    "amex-l1.json",
    "amex-l3.json",
    "gui-odyssey-l3.json",
    "aitw-l1.json",
    "aitw-l2.json",
    "gui-odyssey-l2.json",
]

# TODO: some of the mappings above must be wrong because the conversion fails for some subsets
config_dict = [
    # {
    #     "json_path": "mind2web-l1.json",
    #     "images_folder": "mind2web/",
    #     "sampling_strategy": "all",
    # },
    # {
    #     "json_path": "mind2web-l2.json",
    #     "images_folder": "mind2web/",
    #     "sampling_strategy": "all",
    # },
    {
        "json_path": "mind2web-l3.json",
        "images_folder": "mind2web/",
        "sampling_strategy": "all",
    },
    {
        "json_path": "guiact-web-single.json",
        "images_folder": "guiact-web-single/images/",
        "sampling_strategy": "all",
    },
    # {
    #     "json_path": "guiact-web-multi-l1.json",
    #     "images_folder": "guiact-web-multi-v2/images",
    #     "sampling_strategy": "all",
    # },
    {
        "json_path": "guiact-web-multi-l3.json",
        "images_folder": "guiact-web-multi-v2/images",
        "sampling_strategy": "all",
    },
    # {
    #     "json_path": "miniwob-l1.json",
    #     "images_folder": "images",
    #     "sampling_strategy": "all",
    # },
    {
        "json_path": "miniwob-l3.json",
        "images_folder": "images",
        "sampling_strategy": "all",
    },
    {
        "json_path": "coat.json",
        "images_folder": "coat/images/",
        "sampling_strategy": "all",
    },
    {
        "json_path": "android_control.json",
        "images_folder": "android_control/images/",
        "sampling_strategy": "all",
    },
    # {
    #     "json_path": "gui-odyssey-l1.json",
    #     "images_folder": "gui-odyssey/images/",
    #     "sampling_strategy": "random:33%",
    # },
    # {
    #     "json_path": "gui-odyssey-l2.json",
    #     "images_folder": "gui-odyssey/images/",
    #     "sampling_strategy": "random:33%",
    # },
    {
        "json_path": "gui-odyssey-l3.json",
        "images_folder": "gui-odyssey/images/",
        "sampling_strategy": "random:33%",
    },
    # {
    #     "json_path": "amex-l1.json",
    #     "images_folder": "amex/images/",
    #     "sampling_strategy": "random:33%",
    # },
    # {
    #     "json_path": "amex-l2.json",
    #     "images_folder": "amex/images/",
    #     "sampling_strategy": "random:33%",
    # },
    {
        "json_path": "amex-l3.json",
        "images_folder": "amex/images/",
        "sampling_strategy": "random:33%",
    },
    # {
    #     "json_path": "aitw-l1.json",
    #     "images_folder": "aitw-v1/images",
    #     "sampling_strategy": "all",
    # },
    {
        "json_path": "aitw-l3.json",
        "images_folder": "aitw-v1/images/",
        "sampling_strategy": "all",
    },
]


def authenticate_huggingface():
    """Authenticate with HuggingFace Hub using token."""
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("Authenticating with HuggingFace Hub using token...")
        login(token=hf_token)
    else:
        raise ValueError("HF_TOKEN environment variable not set.")


def discover_dataset_config(dataset_path: str) -> List[Dict[str, Any]]:
    """Discover dataset configuration by scanning the data directory."""
    dataset_dir = Path(dataset_path)
    train_dir = dataset_dir

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    configs = []
    processed_splits = set()

    # Find all JSON files in the train directory
    for config in config_dict:
        subset_name = (
            config["json_path"]
            .replace(".json", "")
            .replace("-l1", "")
            .replace("-l2", "")
            .replace("-l3", "")
        )

        # Skip if we already processed this split
        if subset_name in processed_splits:
            continue

        config["subset_name"] = subset_name
        configs.append(config)
        processed_splits.add(subset_name)
        print(
            f"Discovered config: {config['subset_name']} -> {config['images_folder']}"
        )

    return configs


def download_dataset(
    repo_id: str = "xlangai/aguvis-stage2", local_dir: str = "./aguvis_raw"
) -> str:
    """Download the dataset using snapshot_download."""
    print(f"Downloading dataset from {repo_id}...")
    local_path = snapshot_download(
        repo_id=repo_id, local_dir=local_dir, repo_type="dataset"
    )
    print(f"Dataset downloaded to: {local_path}")
    return local_path


def extract_zip_files(dataset_path: str):
    """Extract all zip files found in the dataset directory, but only if not already extracted."""
    print("Extracting zip files...")
    dataset_dir = Path(dataset_path)

    for zip_file in dataset_dir.rglob("*.zip"):
        extract_dir = zip_file.parent / zip_file.stem
        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(
                f"Skipping extraction for {zip_file} (already extracted at {extract_dir})"
            )
            continue

        print(f"Extracting: {zip_file}")
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extracted to: {extract_dir}")


def extract_tar_parts_grouped(dataset_path: str):
    """
    Finds all .tar.gz.part_* groups, merges them, and extracts them into directories
    named after their common prefix.
    """
    dataset_dir = Path(dataset_path)
    part_files = list(dataset_dir.glob("*.tar.gz.part_*"))

    if not part_files:
        print("No split .tar.gz.part_* files found.")
        return

    # Group part files by prefix
    groups = defaultdict(list)
    for part in part_files:
        prefix = part.name.split(".tar.gz.part_")[0]
        groups[prefix].append(part)

    for prefix, parts in groups.items():
        parts = sorted(parts)  # Ensure correct order
        merged_tar_path = dataset_dir / f"{prefix}.tar.gz"
        extract_dir = dataset_dir / prefix

        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(
                f"Skipping extraction for '{prefix}' (already extracted at {extract_dir})"
            )
            continue

        # Merge parts
        CHUNK_SIZE = 1024 * 1024
        print(f"Merging parts for '{prefix}'...")
        with open(merged_tar_path, "wb") as outfile:
            for part in parts:
                print(f"  Adding: {part.name}")
                with open(part, "rb") as infile:
                    while chunk := infile.read(CHUNK_SIZE):
                        outfile.write(chunk)

        print(f"Merged to: {merged_tar_path}")

        # Extract
        print(f"Extracting to: {extract_dir}")
        with tarfile.open(merged_tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        print(f"Done extracting '{prefix}'\n")


def check_subset_exists(repo_id: str, subset_name: str) -> bool:
    """Check if a subset already exists in the remote dataset."""
    try:
        # Try to get dataset info with specific subset
        from datasets import get_dataset_config_names

        config_names = get_dataset_config_names(repo_id)
        return subset_name in config_names
    except Exception as e:
        print(f"Could not check if subset exists: {e}")
        return False


def load_image_from_folder(images_folder: Path, img_path: str) -> Image.Image:
    """Load images from the specified folder."""
    full_path = images_folder / img_path
    img = Image.open(full_path)
    return img


def convert_to_code_agent_format(messages: list[ChatMessage], json_path: str):
    for i, message in enumerate(messages):
        content = message.content

        if message.role == "system":
            if json_path in MOBILE_FILE:
                content = MOBILE_SYSTEM_PROMPT
            else:
                content = OS_SYSTEM_PROMPT

        if message.role == "user":
            content = content.replace("<image>", "")

        elif message.role == "assistant":
            content = (
                content.replace("Action: ", "")
                .replace("Observation: ", "")
                .replace("Thought: ", "")
            )
            if i == len(messages) - 1:
                content = (
                    "<code>\n" + content.strip() + "\n</code>"
                )
            else:
                # TODO: Check if there is always only 2 assistants
                content = (
                    "<think>\n"
                    + content.strip()
                    + "\n</think>\n"
                )

        messages[i].content = content

        # Fuse subsequent messages have the same role, merge it
        if i > 0 and messages[i].role == messages[i - 1].role:
            # Need to fuse both messages
            messages[i - 1].content += messages[i].content
            messages.pop(i)

    return messages


def convert_to_chat_format(
    data: ConversationData, json_path: str
) -> list[ChatMessage]:
    """Convert data item to chat template format."""
    # This is a placeholder - you'll need to adapt this based on the actual data structure
    # The exact conversion depends on how the original data is structured
    chat_messages = data.to_chat_messages()
    chat_messages = convert_to_code_agent_format(chat_messages, json_path)
    return chat_messages


def convert_to_new_action_space(
    messages: list[ChatMessage], resolution: tuple[int, int]
) -> list[ChatMessage]:
    regex_match = None
    index = -1
    regex = r"<code>\n(.*?)\n</code>"
    assistant_msg = [message for message in messages if message.role == "assistant"]
    if assistant_msg:
        for i, msg in enumerate(assistant_msg):
            regex_match = re.search(regex, msg.content, re.DOTALL)
            index = i
            if regex_match is not None:
                break
        if regex_match is not None:
            function_calls = parse_function_call(
                regex_match.group(1),
                pattern_to_match=["pyautogui", "mobile", "terminate", "answer"],
            )
            
            if len(function_calls) > 0:

                for i, function_call in enumerate(deepcopy(function_calls)):
                    
                    if function_call.function_name == "pyautogui.dragTo":
                        x1, y1 = function_calls[i-1].parameters.values()
                        x2, y2 = function_calls[i].parameters.values()
                        function_calls[i].parameters = {"from_coord": (x1, y1), "to_coord": (x2, y2)}
                        function_calls[i].original_string = function_calls[i].to_string()
                        function_calls.pop(i-1)

                function_calls = action_conversion(function_calls, resolution=resolution)

                new_action_string = "\n".join(
                    [function_call.to_string() for function_call in function_calls]
                )
                assistant_msg[index].content = assistant_msg[index].content.replace(
                    regex_match.group(1), new_action_string
                )

    return messages


def process_subset(
    config: Dict[str, Any],
    dataset_path: str,
) -> tuple[ConversationDataList, Path]:
    """Process a single dataset subset."""
    subset_name = config["subset_name"]

    print(f"Processing split: {subset_name}")

    dataset_dir = Path(dataset_path)
    images_folder = dataset_dir / config["subset_name"] / config["images_folder"]

    if not images_folder.exists():
        print(f"Images folder not found: {images_folder}")
    else:
        print(f"Images folder: {images_folder}")

    json_config_path = dataset_dir / config["json_path"]
    with open(json_config_path, "r") as f:
        data = ConversationDataList.model_validate_json(f.read())
        # data = f.read()
        print(f"Added '{json_config_path}'")

    return data, images_folder


def row_generator(
    data: ConversationDataList, images_folder: Path, json_path: str
) -> Generator[Dict[str, Any], None, None]:
    conversations: list[ConversationData] = data.root
    for item in tqdm(conversations):
        # Extract image paths from the data item
        try:
            # Load images
            image = load_image_from_folder(images_folder, item.image)
            chat_message = convert_to_chat_format(item, json_path)
            chat_message = convert_to_new_action_space(chat_message, image.size)
            if len(chat_message) == 0:
                continue

            row = DataRow.from_chat_messages(chat_message, image)
            yield row.model_dump()
        except Exception as e:
            print(f"Error processing item: {e}", item)
            continue


def make_dataset_from_original_data():
    """Main function to orchestrate the entire process."""
    load_dotenv(override=True)

    print("Starting aguvis-stage2 dataset processing...")

    # Step 0: Authenticate with HuggingFace Hub
    authenticate_huggingface()

    dataset_path = download_dataset(
        "xlangai/aguvis-stage2", "/fsx/amir_mahla/aguvis_raw"
    )

    # extract_zip_files(dataset_path)
    # extract_tar_parts_grouped(dataset_path)

    dataset_configs = discover_dataset_config(dataset_path)
    converted_folder = "/fsx/amir_mahla/aguvis_converted"
    os.makedirs(converted_folder, exist_ok=True)
    converted_repo_id = "smolagents/aguvis-stage-2"

    # TODO: Make it in multi processing
    for config in dataset_configs:
        print(f"\n{'=' * 50}")
        print(config)

        # # Check if the subset already exists in the remote dataset
        subset_name = config["subset_name"]
        if check_subset_exists(converted_repo_id, subset_name):
            print(
                f"Subset '{subset_name}' already exists in {converted_repo_id}, skipping processing."
            )
            continue
        json_path = config["json_path"]
        data, image_folder = process_subset(
            config, dataset_path
        )


        print("Creating dataset...")
        # Collect all rows first
        rows = []
        for row in row_generator(data, image_folder, json_path):
            rows.append(row)
        
        # Create dataset from collected data
        data = Dataset.from_list(rows)
        

        # print("Pushing to hub...")
        # # Fix: Use config_name for subset name and split="train"
        data.push_to_hub(
            "smolagents/aguvis-stage-2",
            config_name=config["subset_name"],  # This sets the subset name
            split="train",  # This should be "train" not the subset name
        )

        # print(f"Processed and uploaded subset: {config['subset_name']}")

        # # Force garbage collection to manage memory
        gc.collect()

#     # Cleanup
#     print("\nCleaning up temporary files...")
#     # shutil.rmtree(dataset_path, ignore_errors=True)
#
    api.upload_large_folder(folder_path=converted_folder, repo_id="smolagents/aguvis-stage-2", repo_type="dataset")
#
#     shutil.rmtree(converted_folder, ignore_errors=True)
#
#     print("All done!")


if __name__ == "__main__":
     #     print(dataset)
 
     #     dataset = dataset.map(change_coordinates, num_proc=32)
 
     #     dataset.push_to_hub("smolagents/aguvis-stage-2", subset, split="train")
     make_dataset_from_original_data()
