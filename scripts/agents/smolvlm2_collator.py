from PIL import Image
from scripts.agents.function_parser import parse_function_call
import numpy as np
from transformers.models.smolvlm.image_processing_smolvlm import get_resize_output_image_size
from transformers.image_utils import ChannelDimension

def resize_images_in_messages(batch_messages) -> list[Image.Image]:

    all_image_inputs = []
    for messages in batch_messages:

        old_image = messages[1]["content"][0]["image"]

        resized_height, resized_width = get_resize_output_image_size(
                np.array(old_image), resolution_max_side=3*384, input_data_format=ChannelDimension.LAST
            )
        new_image = old_image.resize((resized_width, resized_height))
        messages[1]["content"][0]["image"] = new_image

        function_calls = parse_function_call(messages[2]["content"][0]["text"])
        old_function_call_strings = [
            function_call.to_string() for function_call in function_calls
        ]
        for function_call, old_function_call_string in zip(function_calls, old_function_call_strings):
            if function_call.function_name in [
                "click",
                "long_press",
                "double_click",
                "move_mouse",
            ]:
                function_call.parameters["arg_0"] = (
                    int(function_call.parameters["arg_0"]
                    / old_image.width
                    * new_image.width)
                )
                function_call.parameters["arg_1"] = (
                    int(function_call.parameters["arg_1"]
                    / old_image.height
                    * new_image.height)
                )
            elif function_call.function_name in ["swipe", "drag"]:
                function_call.parameters["arg_0"] = (
                    int(function_call.parameters["arg_0"][0]
                    / old_image.width
                    * new_image.width),
                    int(function_call.parameters["arg_0"][1]
                    / old_image.height
                    * new_image.height)
                )
                function_call.parameters["arg_1"] = (
                    int(function_call.parameters["arg_1"][0]
                    / old_image.width
                    * new_image.width),
                    int(function_call.parameters["arg_1"][1]
                    / old_image.height
                    * new_image.height)
                )
            messages[2]["content"][0]["text"] = messages[2]["content"][0]["text"].replace(old_function_call_string, function_call.to_string())


        all_image_inputs.append([new_image])
    return all_image_inputs

def create_vlm_collate_fn(processor, script_args):
    """Optimized collate function for VLM training that masks system prompt tokens."""

    def collate_fn(examples: dict[str, list | str | Image.Image]):
        batch_messages: list[list[dict[str, list | str | Image.Image]]] = []
        system_prompts: list[str] = []
        user_prompts: list[list[str]] = []
        all_image_inputs: list[list[Image.Image]] = []
        for example in examples:

            images = example["images"]
            users = []

            for text in example["texts"]:

                sample = []
                if "system" in text:
                    system = "\n" + text["system"]
                    sample.append({"role": "system", "content": [{"type": "text", "text": system}]})
                    system_prompts.append(system)


                user = "\n" + text["user"]
                users.append(user)
                if not sample:
                    sample.append({"role": "user", "content": [
                                    {"type": "image", "image": images[0]},
                                    {"type": "text", "text": user},
                                ]})
                else:
                    sample.append({"role": "user", "content": [
                                    {"type": "text", "text": user},
                                ]})

                assistant = text["assistant"]
                sample.append({"role": "assistant", "content": [{"type": "text", "text": assistant}]})

            system_prompts.append(system)
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

        system_encodings = processor.tokenizer(
            system_prompts, add_special_tokens=True, padding=False
        )["input_ids"]

        user_encodings = processor.tokenizer(
            user_prompts, add_special_tokens=False, padding=False
        )["input_ids"]

        for sample in [system_encodings, user_encodings]:
            encodings = list()
            for i, system_ids in enumerate(encodings):
                if input_ids[i, : len(system_ids)].tolist() == system_ids:
                    labels[i, : len(system_ids)] = -100
                else:
                    seq = input_ids[i].tolist()
                    for j in range(len(seq) - len(system_ids) + 1):
                        if seq[j : j + len(system_ids)] == system_ids:
                            labels[i, j : j + len(system_ids)] = -100
                            break  # early exit

        batch["labels"] = labels
        return batch

    return collate_fn
