import logging

from typing import Union

import jinja2
import jinja2.exceptions

logger = logging.getLogger("sillm")

default_templates = {
    "llama": "llama2",
    "mistral": "mistral",
    "gemma": "gemma",
    "mixtral": "mistral",
    "phi": "phi2",
    "qwen2": "qwen2"
}

class Template(object):
    def __init__(self,
                 template: str = "chatml"
                 ):
        loader = jinja2.PackageLoader('sillm', 'templates')
        env = jinja2.Environment(loader=loader, trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = self._raise_exception

        fname_template = template + ".jinja"
        self.template = env.get_template(fname_template)

    def _raise_exception(self, msg):
        raise jinja2.exceptions.TemplateError(msg)

    def apply_chat_template(self,
                            messages: list,
                            add_generation_prompt: bool = False
                            ):
        template_args = {
            "add_generation_prompt": add_generation_prompt
        }
        
        return self.template.render(messages=messages, **template_args)
    
    @staticmethod
    def from_args(self,
                  args
                  ):
        if args.model_type and args.model_type in default_templates:
            template_name = default_templates[args.model_type]
            
            return Template(template_name)

        return None

class Conversation(object):
    """
    Wrapper for conversation templates.
    """
    def __init__(self,
                template: Template,
                system_prompt: str = None):
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

    def apply_chat_template(self,
                            add_generation_prompt: bool = False
                            ):
        return self.tokenizer.apply_chat_template(self.messages, tokenize=False)