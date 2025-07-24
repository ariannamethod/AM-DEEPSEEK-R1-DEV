from typing import List, Optional, Literal
from pydantic import BaseModel, Field, RootModel, field_validator
from copy import deepcopy
from PIL import Image


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

    @staticmethod
    def from_conversation_list(data: list[dict[str, str]]) -> list["ChatMessage"]:
        messages = []
        for item in data:
            if item["from"] == "system":
                role = "system"
            elif item["from"] == "human":
                role = "user"
            else:
                role = "assistant"
            messages.append(ChatMessage(role=role, content=item["value"]))
        return messages


class ConversationEntry(BaseModel):
    from_: Literal["system", "human", "gpt"]  = Field(alias="from")
    value: str
    recipient: Optional[str] = None
    end_turn: Optional[bool] = None

    def to_chat_message(self) -> ChatMessage:
        if self.from_ == "system":
            role: Literal["user", "assistant", "system"] = "system"
        elif self.from_ == "human":
            role = "user"
        else:
            role = "assistant"
        return ChatMessage(role=role, content=self.value)


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

class DataRow(BaseModel):
    system: str 
    user: str
    assistant: str
    image: Image.Image

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_chat_messages(cls, messages: list[ChatMessage], image: Image.Image) -> "DataRow":
        system, user, assistant = None, None, None
        for message in messages:
            if message.role == "system":
                system = message.content
            elif message.role == "user":
                user = message.content
            elif message.role == "assistant":
                assistant = message.content

        return cls(system=system, user=user, assistant=assistant, image=image) # type: ignore

    def to_model_dump(self) -> dict:
        return {
            "system": self.system,
            "user": self.user,
            "assistant": self.assistant,
            "image": self.image,
        }