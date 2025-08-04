from PIL import Image
from scripts.agents.function_parser import parse_function_call
import json
from scripts.agents.smolvlm2_collator import create_vlm_collate_fn
from scripts.agents.models import ConversationDataList, ConversationData
from scripts.agents.config import config_dict_stage_1, config_dict_stage_2
from transformers import AutoProcessor
from datasets import load_dataset


def read_json(path: str) -> list[ConversationData]:
    with open(path, "r") as f:
        data = ConversationDataList.model_validate_json(f.read())
    return data.root

def check_duplicate_images(data: list[ConversationData], json_path: str) -> int:
    images_set_path = set()
    images_path = []
    for example in data:

        images_set_path.add(example.image)
        images_path.append(example.image)

    print()
    if len(images_set_path) != len(images_path):
        print(f"Duplicate images in {json_path}. difference: {len(images_path) - len(images_set_path)}")
        print(f"len(images_path): {len(images_path)}")
        print(f"len(images_set_path): {len(images_set_path)}")
        print(f"user/assistant by image: {(len(images_path) - len(images_set_path)) / len(images_set_path)}")
    else:
        print(f"No duplicate images in {json_path}")
        print(f"len(images_path): {len(images_path)}")
        print(f"len(images_set_path): {len(images_set_path)}")
        print(f"user/assistant by image: {(len(images_path) - len(images_set_path)) / len(images_set_path)}")
    print()

    return len(images_path) - len(images_set_path), len(images_path) 


def get_function_call_names(
    data: ConversationData,
    json_path: str,
    has_os_action_space: bool,
    has_mobile_action_space: bool,
) -> tuple[set[str], bool, bool]:
    function_call_names = set()
    token_count = 0
    for message in data.conversations:
        if message.from_ == "gpt":
            function_calls = parse_function_call(message.value)
            if not has_os_action_space and any(
                function_call.function_name.startswith("pyautogui")
                for function_call in function_calls
            ):
                has_os_action_space = True
                print(f"{json_path} has OS action space")
            if not has_mobile_action_space and any(
                function_call.function_name.startswith("mobile")
                for function_call in function_calls
            ):
                has_mobile_action_space = True
                print(f"{json_path} has Mobile action space")
            for function_call in function_calls:
                function_call_names.add(function_call.function_name)
    return function_call_names, has_os_action_space, has_mobile_action_space


def get_token_count(data: dict, processor: AutoProcessor) -> int:
    token_count = 0
    for message in data["texts"]:
        token_count += len(
            processor.tokenizer(message["assistant"], add_special_tokens=False)[
                "input_ids"
            ]
        )
    return token_count


# def create_vlm_collate_fn(processor):
#     """Optimized collate function for VLM training that masks system prompt tokens."""
#
#     def collate_fn(example: dict[str, list | str | Image.Image]):
#         batch_messages = []
#         system_prompts = []
#         user_prompts = []
#         assistant_messages = []
#         chat_messages = []
#         all_image_inputs = example["images"]
#
#         for text in example["texts"]:
#
#             sample = []
#             if "system" in text:
#                 system = "\n" + text["system"]
#                 sample.append({"role": "system", "content": [{"type": "text", "text": system}]})
#
#
#             user = "\n" + text["user"]
#             if not chat_messages:
#                 sample.append({"role": "user", "content": [
#                                 {"type": "image", "image": all_image_inputs[0]},
#                                 {"type": "text", "text": user},
#                             ]})
#             else:
#                 sample.append({"role": "user", "content": [
#                                 {"type": "text", "text": user},
#                             ]})
#
#             assistant = text["assistant"]
#             sample.append({"role": "assistant", "content": [{"type": "text", "text": assistant}]})
#
#             assistant_messages.append(assistant)
#
#
#         texts = [processor.apply_chat_template(
#                 chat_messages, tokenize=False, add_generation_prompt=False
#             )]
#
#         batch = processor(
#             text=texts,
#             images=all_image_inputs if all_image_inputs else None,
#             padding=True,
#             return_tensors="pt",
#             max_length=4096,
#         )
#
#         assistant_encodings = processor.tokenizer(
#             assistant_messages, add_special_tokens=False, padding=False
#         )["input_ids"]
#
#         return batch["input_ids"][0], assistant_encodings[0]
#
#     return collate_fn


def create_vlm_collate_fn(processor):
    """Optimized collate function for VLM training that masks system prompt tokens."""

    def collate_fn(examples: list[dict[str, list | str | Image.Image]]):
        batch_messages: list[list[dict[str, list | str | Image.Image]]] = []
        system_prompts: list[str] = []
        user_prompts: list[list[str]] = []
        all_image_inputs: list[list[Image.Image]] = []
        for example in examples:
            images: list[Image.Image] = example["images"]
            users: list[str] = []

            for text in example["texts"]:
                sample: list[dict[str, list | str | Image.Image]] = []
                if "system" in text.keys():
                    system = "\n" + text["system"]
                    sample.append(
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": system}],
                        }
                    )
                    system_prompts.append(system)

                user = "\n" + text["user"]
                if not users:
                    sample.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": images[0]},
                                {"type": "text", "text": user},
                            ],
                        }
                    )
                else:
                    sample.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user},
                            ],
                        }
                    )
                users.append(user)

                sample.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": text["assistant"]}],
                    }
                )

            user_prompts.append(users)
            batch_messages.append(sample)
            all_image_inputs.append(images)

        # all_image_inputs = resize_images_in_messages(batch_messages)

        texts = [
            processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            for messages in batch_messages
        ]
        print(texts)

        batch = processor(
            text=texts,
            images=all_image_inputs if all_image_inputs else None,
            padding=True,
            return_tensors="pt",
            max_length=4096,
        )

        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        if hasattr(processor, "image_token"):
            image_token_id = processor.tokenizer.convert_tokens_to_ids(
                processor.image_token
            )
            if image_token_id is not None:
                labels[labels == image_token_id] = -100
        else:
            raise ValueError("Processor does not have image_token")


        system_encodings = []
        if system_prompts:
            system_encodings = processor.tokenizer(
                system_prompts, add_special_tokens=True, padding=False
            )["input_ids"]

        user_encodings = [
            processor.tokenizer(user_prompt, add_special_tokens=False, padding=False)[
                "input_ids"
            ]
            for user_prompt in user_prompts
        ]

        if system_encodings and len(user_encodings) != len(system_encodings):
            raise ValueError("User and system encodings have different lengths")

        print(user_encodings)


        for i, user_ids_list in enumerate(user_encodings):
            ids_list = []

            if system_encodings:
                ids_list.extend(system_encodings[i])

            for user_ids in user_ids_list:
                ids_list.extend(user_ids)

            print(ids_list)
            exit()
            for ids in ids_list:
                if input_ids[i, : len(ids)].tolist() == ids:
                    labels[i, : len(ids)] = -100
                else:
                    seq = input_ids[i].tolist()
                    for j in range(len(seq) - len(ids) + 1):
                        if seq[j : j + len(ids)] == ids:
                            labels[i, j : j + len(ids)] = -100
                            break  # early exit

        batch["labels"] = labels
        return batch

    return collate_fn


if __name__ == "__main__":
    from transformers import AutoProcessor
    # from datasets import load_dataset, IterableDataset

    # class ScriptArguments:
    #     image_resize = {
    #         "factor": 28,
    #         "min_pixels": 200704,
    #         "max_pixels": 1003520,
    #     }

    # processor = AutoProcessor.from_pretrained(
    #     "HuggingFaceTB/SmolVLM2-2.2B-Instruct", model_revision="main"
    # )
    # # # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", model_revision="main", min_pixels=200704, max_pixels=1003520)
    # collate_fn = create_vlm_collate_fn(processor)
    # max = 0
    # answer_token = 0
    # user_token = 0
    # system_token = 0
    # total_token = 0
    # Aguvis_stage_2
    #    datasets = load_dataset("smolagents/aguvis-stage-2", dataset_name, split="train")
    #    for i, example in enumerate(datasets):
    #        batch = collate_fn([example])
    #        exit()
    # Aguvis_stage_1
    # for dataset_name in ['omniact', 'ricoig16k', 'webui350k', 'widget_captioning', 'seeclick', 'ui_refexp', 'ricosca', 'guienv']:
    function_call_names = set()
    token_count = 0
    assistant_token_count = 0

    for config in config_dict_stage_1:
        dataset_name = config["json_path"].split(".")[0]
        print("Processing: ", dataset_name)
        data: list[ConversationData] = read_json(f"/fsx/amir_mahla/aguvis_raw_stage_1/{dataset_name}.json")
        diff, original_length = check_duplicate_images(data, f"{dataset_name}.json")
        token_count += original_length - diff
        # has_os_action_space = False
        # has_mobile_action_space = False
        # current_dataset_token_count = 0
        # current_dataset_assistant_token_count = 0
        # for example in data:
        #     # token_count += get_token_count(example, processor)
        #     input_ids, assistant_encodings = collate_fn([example])
        #     exit()
        #     assistant_token_count += len(assistant_encodings)
        #     total_token += len(input_ids)
        #     current_dataset_token_count += len(input_ids)
        #     current_dataset_assistant_token_count += len(assistant_encodings)
        #     # function_names, has_os_action_space, has_mobile_action_space = get_function_call_names(example, f"data/aguvis_raw/{dataset_name}.json", has_os_action_space, has_mobile_action_space)
        #     # for function_name in function_names:
        #     #     function_call_names.add(function_name)

        # print(
        #     "Dataset: ",
        #     dataset_name,
        #     "Total token: ",
        #     current_dataset_token_count,
        #     "Total assistant token: ",
        #     current_dataset_assistant_token_count,
        # )

    # print("Total token: ", total_token)
    # print("Total assistant token: ", assistant_token_count)

    # print()
    # for function_call_name in function_call_names:
    #     if function_call_name.startswith("pyautogui") or function_call_name.startswith("mobile"):
    #         print(function_call_name)

    # print()
    # for function_call_name in function_call_names:
    #     print(function_call_name)

    print(token_count)

    # print(batch)
