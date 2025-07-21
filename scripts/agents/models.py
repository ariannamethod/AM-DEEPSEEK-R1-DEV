from typing import List, Optional, Literal
from pydantic import BaseModel, Field, RootModel, field_validator
from copy import deepcopy
from PIL import Image


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ConversationEntry(BaseModel):
    from_: Literal["system", "human", "gpt"]  = Field(alias="from")
    value: str
    recipient: Optional[str] = None
    end_turn: Optional[bool] = None

    def to_chat_message(self) -> ChatMessage
        if self.from_ == "system":
            role = "system"
        elif self.from_ == "human":
            role = "user"
        else:
            role = "assistant"
        return ChatMessage(role=role, value=self.value)


class ConversationData(BaseModel):
    image: str 
    conversations: List[ConversationEntry]

    @field_validator("image", mode="before")
    def validate_image(cls, v):
        if isinstance(v, list):
            if len(v) == 1:
                return v[0]
            elif len(v) == 2:
                return v[1]
            else:
                raise ValueError("Expected 1 or 2 images, got multiple")
        return v

    def to_chat_messages(self) -> list[ChatMessage]:
        return [conversation.to_chat_message() for conversation in self.conversations]

class ConversationDataList(RootModel[List[ConversationData]]):
    pass

class DataRow:
    system: str 
    user: str
    assitant: str
    image: Image.Image

    def from_chat_messages(messages: list[ChatMessage], image: Image.Image) -> "DataRow":
        system, user, assistant = None
        for message in messages:
            if message.role == "system":
                system = message.content
            elif message.role == "user":
                user = message.content
            elif message.role == "assistant":
                assistant = message.content

        return DataRow(system=system, user=user, assistant=assistant, image=image)
            

if __name__ == "__main__":
    from os import listdir
    import PIL.Image as Image
    import re

    def extract_function_name(text):
        """
        Extract function name from a function call, including optional dots for instance methods.

        Examples:
        - cls.name(x, y) -> 'cls.name'
        - function() -> 'function'
        """

        # Regex pattern to match function calls and extract the function name
        # This pattern matches:
        # - Any word characters, optional dots, and underscores before the opening parenthesis
        # - The opening parenthesis and everything after it_raw
        pattern = r'([a-zA-Z_][a-zA-Z0-9_.]*)\('

        # Find all matches
        matches = re.findall(pattern, text)

        for match in deepcopy(matches):
            if "pyautogui" not in match and "mobile" not in match and "terminate" not in match and "answer" not in match:
                matches.remove(match)
        
        # Check if there are multiple matches
        if len(matches) >= 1:
            return matches
        return None

    repertory = "/fsx/amir_mahla/aguvis_raw"
    actions: set[str] = set()
    for file in listdir(repertory):
        if file.endswith(".json"):
            with open(f"{repertory}/{file}", "r") as f:
                try:
                    data = ConversationDataList.model_validate_json(f.read())
                    print(f"Successfully parsed {file}")
                    for conversation in data.root:
                        names = extract_function_name(conversation.conversations[-1].value)
                        if names:
                            for name in names:
                                if "pyautogui" in name or "mobile" in name or "terminate" in name or "answer" in name:
                                    actions.add(name)
                except Exception as e:
                    print(f"Error parsing {file}")
                    raise e
    print()
    print(actions)
    # for action in actions:
    #     if "pyautogui" in action or "mobile" in action or "terminate" in action or "answer" in action or "browser" in action:
    #         print(action)
    # print()
    # for action in actions:
    #     if "pyautogui" not in action and "mobile" not in action and "terminate" not in action and "answer" not in action and "browser" not in action:
    #         print(action)