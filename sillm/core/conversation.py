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
                            add_generation_prompt: bool = False
                            ):
        return self.template.apply_chat_template(messages=messages, add_generation_prompt=add_generation_prompt, tools=self.tools, **self.params)

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
                    add_generation_prompt: bool = False,
                    add_generation_prefix: str = None
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

        # Add system message
        if len(self.messages) == 0 and self.system_prompt is not None:
            msg_system = {
                "role": "system",
                "content": self.system_prompt
            }
            messages.append(msg_system)
            
        # Add message
        msg = {
            "role": role,
            "content": content
        }
        messages.append(msg)

        self.messages += messages

        text = self.apply_chat_template(messages=messages, add_generation_prompt=add_generation_prompt)
        if add_generation_prefix is not None:
            text += add_generation_prefix

        if len(text) == 0:
            raise ValueError("Chat template returned empty string")

        return text

    def add_user(self,
                 content: str,
                 add_generation_prompt: bool = True,
                 add_generation_prefix: str = None
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
        return self.add_message(content, "user", add_generation_prompt=add_generation_prompt, add_generation_prefix=add_generation_prefix)
    
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
    
class AutoConversation(Conversation):
    """
    Wrapper for tokenizers with built-in chat templates.
    """
    def apply_chat_template(self,
                            messages: list,
                            add_generation_prompt: bool = False
                            ):
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt, tools=self.tools, **self.params)