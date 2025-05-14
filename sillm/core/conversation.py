import logging
import json

from sillm.core.template import Template, AutoTemplate

logger = logging.getLogger("sillm")

class Conversation(object):
    """
    Wrapper for conversation templates.
    """
    def __init__(self,
                template: Template,
                system_prompt: str = None,
                tools: list = None,
                params: dict = {}
                ):
        self.template = template
        self.system_prompt = system_prompt
        self.tools = tools
        self.params = params

        self.clear()

    def apply_chat_template(self,
                            messages: list,
                            add_generation_prompt: bool = False,
                            tools: list = None
                            ):
        return self.template.apply_chat_template(messages=messages, add_generation_prompt=add_generation_prompt, tools=tools, **self.params)

    def __str__(self):
        """
        Return formatted string using the template.
        Returns:
            Formatted conversation string.
        """
        return self.apply_chat_template()
        
    def clear(self):
        """
        Clear conversation.
        """
        self.messages = []
        self.text = ""

    def save_json(self,
                  fpath: str
                  ):
        """
        Save conversation to JSON file.
        Args:
            fpath: File path.
        """
        with open(fpath, "w") as f:
            json.dump(self.messages, f)

    def load_json(self,
                  fpath: str
                  ):
        """
        Load conversation from JSON file.
        Args:
            fpath: File path.
        """
        with open(fpath, "r") as f:
            self.messages = json.load(f)

    def add_message(self,
                    content: str,
                    role: str,
                    add_generation_prompt: bool = False
                    ):
        """
        Add message to the conversation.
        Args:
            content: Message content.
            role: Role used for the message.
            add_generation_prompt: Whether to add generation prompt.
            add_generation_prefix: Prefix for generation.
        Returns:
            Formatted string for the message.
        """
        messages = []
        tools = None

        # Add system message
        if len(self.messages) == 0:
            if self.system_prompt is not None:
                msg_system = {
                    "role": "system",
                    "content": self.system_prompt
                }
                messages.append(msg_system)

            tools = self.tools
            
        # Add message
        msg = {
            "role": role,
            "content": content
        }
        messages.append(msg)
        self.messages += messages

        if role == "user":
            text = self.apply_chat_template(messages=messages, add_generation_prompt=add_generation_prompt, tools=tools)
        else:
            text = self.apply_chat_template(messages=self.messages, add_generation_prompt=False, tools=tools)
            text = text[len(self.text):]

        self.text = self.apply_chat_template(messages=self.messages, add_generation_prompt=add_generation_prompt, tools=tools)

        if len(text) == 0:
            raise ValueError("Chat template returned empty string")

        return text

    def add_user(self,
                 content: str,
                 add_generation_prompt: bool = True
                 ):
        """
        Add user message to the conversation.
        Args:
            content: User prompt.
            add_generation_prompt: Whether to add generation prompt.
            add_generation_prefix: Prefix for generation.
        Returns:
            Formatted string for user message.
        """
        return self.add_message(content, "user", add_generation_prompt=add_generation_prompt)
    
    def add_assistant(self,
                     content: str
                     ):
        """
        Add assistant message to the conversation.
        Args:
            content: Assistant response.
        Returns:
            Formatted string for assistant message.
        """
        return self.add_message(content, "assistant")
    
    def add_tool_calls(self,
                       contents: list,
                       role: str = "tool",
                       add_generation_prompt: bool = True
                       ):
        """
        Add tool call results to the conversation.
        """
        for content in contents:
            msg = {
                "role": role,
                "content": content
            }
            self.messages.append(msg)

        text = self.apply_chat_template(messages=self.messages, add_generation_prompt=add_generation_prompt)
        text = text[len(self.text):]

        if len(text) == 0:
            raise ValueError("Chat template returned empty string")
        
        return text
    
class AutoConversation(Conversation):
    """
    Wrapper for tokenizers with built-in chat templates.
    """
    def apply_chat_template(self,
                            messages: list,
                            add_generation_prompt: bool = False,
                            tools: list = None
                            ):
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt, tools=tools, **self.params)