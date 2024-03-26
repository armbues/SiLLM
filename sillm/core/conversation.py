import logging

from sillm.core.template import Template

logger = logging.getLogger("sillm")

class Conversation(object):
    """
    Wrapper for conversation templates.
    """
    def __init__(self,
                template: Template,
                system_prompt: str = None
                ):
        self.template = Template(template)
        self.system_prompt = system_prompt

        self.messages = []

    def apply_chat_template(self,
                            add_generation_prompt: bool = False
                            ):
        return self.template.apply_chat_template(messages=self.messages, add_generation_prompt=add_generation_prompt)

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

    def load_messages(self,
                      messages: list
                      ):
        """
        Load messages into conversation.
        Args:
            messages: List of messages.
        """
        self.messages = messages

    def add_message(self,
                    content: str,
                    role: str,
                    add_generation_prompt: bool = True
                    ):
        if len(self.messages) == 0 and self.system_prompt is not None:
            # Add system message
            msg_system = {
                "role": "system",
                "content": self.system_prompt
            }
            self.messages.append(msg_system)
        msg = {
            "role": role,
            "content": content
        }
        self.messages.append(msg)

        return self.apply_chat_template(add_generation_prompt=add_generation_prompt)

    def add_user(self,
                 content: str,
                 add_generation_prompt: bool = True
                 ):
        """
        Add user message to the conversation.
        Args:
            content: User prompt.
            add_generation_prompt: Whether to add generation prompt.
        Returns:
            Formatted conversation string.
        """
        return self.add_message(content=content, role="user", add_generation_prompt=add_generation_prompt)
    
    def add_assistant(self,
                     content: str
                     ):
        """
        Add assistant message to the conversation.
        Args:
            content: Assistant response.
        Returns:
            Formatted conversation string.
        """
        return self.add_message(content=content, role="assistant")
    
class AutoConversation(Conversation):
    """
    Wrapper for tokenizers with built-in chat templates.
    """
    def __init__(self,
                tokenizer,
                system_prompt: str = None):
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.messages = []

        logger.info(f"Initialized built-in conversation template from tokenizer")

    def apply_chat_template(self,
                            add_generation_prompt: bool = False
                            ):
        return self.tokenizer.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=add_generation_prompt)