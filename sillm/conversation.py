from typing import Union

TEMPLATES = {
    "llama-2": {
        "system":       "[INST] <<SYS>>\n{}\n<</SYS>>\n{}[/INST] ",
        "user":         "[INST]{}[/INST] ",
        "assistant":    "{}\n",
        "stop":         ["[INST]"]
    },
    "chatml": {
        "system":       "<|im_start|>system\n{}<|im_end|>\n",
        "user":         "<|im_start|>user\n{}<|im_end|>\n",
        "assistant":    "<|im_start|>assistant\n{}<|im_end|>\n",
        "stop":         ["<|im_start|>"]
    },
    "alpaca": {
        "system":       "### System Prompt\n{}\n\n### User Message\n{}\n\n",
        "user":         "### User Message\n{}\n\n",
        "assistant":    "### Assistant\n{}\n\n",
        "stop":         ["###"]
    },
    "vicuna": {
        "system":       "{}\nUSER: {}\n",
        "user":         "USER: {}\n",
        "assistant":    "ASSISTANT: {}\n",
        "stop":         ["USER:"]
    },
    "gemma": {
        "system":       "<start_of_turn>user\n{}<end_of_turn>\n", # See https://ai.google.dev/gemma/docs/formatting#system-instructions
        "user":         "<start_of_turn>user\n{}<end_of_turn>\n",
        "assistant":    "<start_of_turn>model\n{}<end_of_turn>\n",
        "stop":         ["<start_of_turn>"]
    }
}

def format_message(content: Union[str, list],
                   template: Union[str, dict] = "llama-2",
                   role: str = "user",
                   strip: bool = False
                   ):
    """
    Format message using conversation template.
    """
    if isinstance(template, str) and template in TEMPLATES:
        template = TEMPLATES[template]
    elif not isinstance(template, dict):
        raise ValueError(f"Template could not be loaded")
    
    if role in template:
        template_format = template[role]
    else:
        raise ValueError(f"Role {role} not found in template")
    
    if role == "system":
        result = template_format.format(*content)
    else:
        result = template_format.format(content)

    if strip:
        return result.rstrip()
    return result

class Conversation(object):
    """
    Wrapper for conversations.
    """
    def __init__(self,
                template: str = "llama-2",
                system_prompt: str = None):
        if type(template) == dict:
            self.template = template
        elif type(template) == str and template in TEMPLATES:
            self.template = TEMPLATES[template]
        else:
            raise ValueError(f"Template could not be loaded")

        self.messages = []
        self.system_prompt = system_prompt
        self._text = ""

        self.trigger = self.template["assistant"].split("{}")[0]

    def __str__(self):
        """
        Return formatted string using the template.
        Returns:
            Formatted conversation string.
        """
        return self._text
    
    def format(self,
               content: Union[str, list],
               role: str = "user"
               ):
            """
            Format message using conversation template.
            Args:
                content: Message content.
                role: Message role.
            Returns:
                Formatted conversation string.
            """
            if role in self.template:
                template_format = self.template[role]
            else:
                raise ValueError(f"Role {role} not found in template")

            if role == "system":
                return template_format.format(*content)

            return template_format.format(content)

    def add_prompt(self,
                   content: str,
                   trigger: bool = True
                   ):
        """
        Add user message to the conversation.
        Args:
            content: User prompt.
        Returns:
            Formatted conversation string.
        """
        if len(self.messages) == 0 and self.system_prompt is not None:
            # Add system message
            msg_system = {
                "role": "system",
                "content": self.system_prompt
            }
            self.messages.append(msg_system)

            msg_user = {
                "role": "user",
                "content": content
            }
            self.messages.append(msg_user)

            self._text += self.format((self.system_prompt, content), role="system")
        else:
            # Add user message
            msg = {
                "role": "user",
                "content": content
            }
            self.messages.append(msg)

            self._text += self.format(content, role="user")

        if trigger:
            return self._text + self.trigger
    
    def add_response(self,
                     content: str
                     ):
        """
        Add assistant message to the conversation.
        Args:
            content: Assistant response.
        Returns:
            Formatted conversation string.
        """
        msg = {
            "role": "assistant",
            "content": content
        }
        self.messages.append(msg)

        self._text += self.format(content, role="assistant")

        return self._text
    
    def clear(self):
        """
        Clear conversation.
        """
        self.messages = []
        self._text = ""