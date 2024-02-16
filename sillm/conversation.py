TEMPLATES = {
    "llama-2": {
        "system":       "[INST] <<SYS>>\n{0}\n<</SYS>>\n\n",
        "user":         "[INST]{0}[/INST]\n",
        "initial":      "{0}[/INST]\n",
        "assistant":    ""
    },
    "chatml": {
        "system":       "<|im_start|>system\n{0}<|im_end|>\n",
        "user":         "<|im_start|>user\n{0}<|im_end|>\n",
        "assistant":    "<|im_start|>assistant\n"
    },
    "alpaca": {
        "system":       "### System Prompt\n{0}\n\n",
        "user":         "### User Message\n{0}\n\n",
        "assistant":    "### Assistant\n"
    }
}

class Conversation(object):
    """
    Wrapper for conversations.
    """
    def __init__(self,
                template: str = "llama-2",
                system: str = None):
        if type(template) == dict:
            self.template = template
        elif type(template) == str and template in TEMPLATES:
            self.template = TEMPLATES[template]
        else:
            raise ValueError(f"Template could not be loaded")

        self.messages = []
        self.system = system
        self._text = ""

    def __str__(self):
        """
        Return formatted string using the template.
        Returns:
            Formatted conversation string.
        """
        return self._text

    def format(self,
               msg: dict
               ):
        """
        Format chat message using conversation template.
        Args:
            msg: Chat message.
        Returns:
            Formatted message string.
        """
        role = msg["role"]
        template_format = self.template[role]
        
        return template_format.format(msg["content"])

    def add_prompt(self,
            content: str
            ):
        """
        Add user message to the conversation.
        Args:
            content: User prompt.
        Returns:
            Formatted conversation string.
        """
        role = "user"

        if len(self.messages) == 0:
            # Add system message
            if self.system is not None:
                msg_system = {
                    "role": "system",
                    "content": self.system
                }
                self._text += self.format(msg_system)
                self.messages.append(msg_system)

                if "initial" in self.template:
                    role = "initial"
        else:
            # Check for consecutive user messages
            if self.messages[-1]["role"] == "user":
                raise ValueError(f"Consecutive user messages")

        # Add user message
        msg = {
            "role": role,
            "content": content
        }
        self._text += self.format(msg)
        if role == "initial":
            msg["role"] = "user"
        self.messages.append(msg)

        # Add prefix for assistant
        self._text += self.template["assistant"]

        return self._text
    
    def add_response(self,
                 content: str,
                 newline: bool = True
                 ):
        """
        Add assistant message to the conversation.
        Args:
            content: Assistant response.
        Returns:
            Formatted conversation string.
        """
        # Add newline
        if not content.endswith("\n") and newline:
            content += "\n"

        msg = {
            "role": "assistant",
            "content": content
        }
        self.messages.append(msg)
        self._text += content

        return self._text
    
    def clear(self):
        """
        Clear conversation.
        """
        self.messages = []
        self._text = ""