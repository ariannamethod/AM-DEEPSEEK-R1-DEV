from typing import List, Optional, Literal
from pydantic import BaseModel, Field, RootModel, field_validator, model_validator
from copy import deepcopy
from PIL import Image
from collections import OrderedDict

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

    @staticmethod
    def from_conversation_list(data: list[dict[str, str]]) -> list["ChatMessage"]:
        messages = []
        system_added = False
        for item in data:
            if item["from"] == "system":
                if not system_added:
                    role: Literal["user", "assistant", "system"] = "system"
                    messages.append(ChatMessage(role=role, content=item["value"]))
                    system_added = True
            elif item["from"] == "human":
                role = "user"
                messages.append(ChatMessage(role=role, content=item["value"]))
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
    recipient: Optional[str] = None
    end_turn: Optional[bool] = None

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
    @model_validator(mode="after")
    def validate_conversation(self):
        max_conversations = 0
        new_conversations: dict[str, ConversationData] = {}

        for i in range(len(self.root) - 1):
            for j in range(i + 1, len(self.root)):
                if self.root[i].image == self.root[j].image:
                    if [c1.model_dump() for c1 in self.root[i].conversations] == [c2.model_dump() for c2 in self.root[j].conversations]:
                        raise ValueError(f"Conversation {self.root[i].conversations} already exists")

        for conversation in self.root:
            if conversation.image not in new_conversations:
                new_conversations[conversation.image] = conversation
            else:
                new_conversations[conversation.image].conversations.extend(conversation.conversations)

                if len(new_conversations[conversation.image].conversations) > max_conversations:
                    max_conversations = len(new_conversations[conversation.image].conversations)
        print(f"Max conversations by image: {max_conversations} conversations")
        self.root = list(new_conversations.values())
        return self

class DataRow(BaseModel):
    images: list[Image.Image]
    texts: list[OrderedDict[str, str]]
    source: str

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_chat_messages(cls, messages: list[ChatMessage], image: Image.Image, source: str) -> "DataRow":
        
        system, user, assistant = None, None, None
        have_system = any(message.role == "system" for message in messages)
        texts: list[OrderedDict[str, str]] = []
        images = [image]
        chat_messages: OrderedDict[str, str] = OrderedDict()
        for message in messages:
            if message.role == "system":
                system = message.content
            elif message.role == "user":
                user = message.content
            elif message.role == "assistant":
                assistant = message.content

            if have_system and user is not None and assistant is not None and system is not None:
                chat_messages["system"] = system
                chat_messages["user"] = user
                chat_messages["assistant"] = assistant
                texts.append(chat_messages)
                chat_messages = OrderedDict()
                user, assistant = None, None

            elif not have_system and user is not None and assistant is not None:
                chat_messages["user"] = user
                chat_messages["assistant"] = assistant
                texts.append(chat_messages)
                chat_messages = OrderedDict()
                user, assistant = None, None

        return cls(images=images, texts=texts, source=source)

    def to_model_dump(self) -> dict:
        return {
            "images": self.images,
            "texts": self.texts,
            "source": self.source,
        }