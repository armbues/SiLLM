import os
import logging

import jinja2
import jinja2.exceptions

logger = logging.getLogger("sillm")

default_templates = {
    "llama": "llama2",
    "mistral": "mistral",
    "gemma": "gemma",
    "gemma2": "gemma",
    "mixtral": "mistral",
    "phi": "phi2",
    "qwen2": "qwen2",
    "dbrx": "chatml",
    "cohere": "cohere",
    "phi3": "phi3",
    "phimoe": "phi3.5",
    "nemotron": "nemotron",
    "pharia-v1": "llama3",
    "granite": "granite",
}

class Template(object):
    def __init__(self,
                 tokenizer,
                 template_name: str = "chatml",
                 exception_callback = None
                 ):
        # Initialize Jinja2 environment
        loader = jinja2.PackageLoader('sillm', 'templates')
        env = jinja2.Environment(loader=loader, trim_blocks=True, lstrip_blocks=True)

        # Set exception callback
        if exception_callback is None:
            env.globals["raise_exception"] = self._raise_exception
        else:
            env.globals["raise_exception"] = exception_callback

        # Set special tokens map
        self.special_tokens_map = tokenizer.special_tokens_map

        if os.path.isfile(template_name):
            with open(template_name, "r") as f:
                template_data = f.read()
                self.template = env.from_string(template_data)
        else:
            fname_template = template_name + ".jinja"
            self.template = env.get_template(fname_template)

        self._name = template_name

        logger.info(f"Initialized conversation template: {template_name}")

    @staticmethod
    def list_templates():
        loader = jinja2.PackageLoader('sillm', 'templates')

        return loader.list_templates()
    
    @staticmethod
    def guess_template(args):
        if args.model_type and args.model_type in default_templates:
            template_name = default_templates[args.model_type]

            # Hack for Llama-3
            if args.model_type == "llama" and args.vocab_size == 128256:
                template_name = "llama3"

            return template_name

        return None

    def _raise_exception(self, msg):
        raise jinja2.exceptions.TemplateError(msg)

    def apply_chat_template(self,
                            messages: list,
                            add_generation_prompt: bool = False,
                            tools: dict = None,
                            ):
        return self.template.render(messages=messages, add_generation_prompt=add_generation_prompt, tools=tools, **self.special_tokens_map)
    
class AutoTemplate(Template):
    def __init__(self,
                 tokenizer
                 ):
        self.tokenizer = tokenizer
        self._name = "auto"

        logger.info(f"Initialized built-in conversation template from tokenizer")
    
    def apply_chat_template(self,
                            messages: list,
                            add_generation_prompt: bool = False,
                            tools: dict = None,
                            ):
        return self.tokenizer.apply_chat_template(messages, tokenize=False, tools=tools, add_generation_prompt=add_generation_prompt)

def init_template(tokenizer,
                  args,
                  template_name: str = None
                  ):
    if template_name is not None:
        # Template name is provided
        if template_name == "auto" and tokenizer.has_template:
            template = AutoTemplate(tokenizer)
        else:
            template = Template(tokenizer, template_name=template_name)
    elif tokenizer.has_template:
            # Use built-in template
            template = AutoTemplate(tokenizer)
    else:
        # Guess template
        template_name = Template.guess_template(args)
        if template_name:
            template = Template(tokenizer, template_name=template_name)
        else:
            # Fall back to empty template
            template = Template(tokenizer, template_name="empty")

            logger.warning("No conversation template found - falling back to empty template.")

    return template