import logging

import jinja2
import jinja2.exceptions

logger = logging.getLogger("sillm")

default_templates = {
    "llama": "llama2",
    "mistral": "mistral",
    "gemma": "gemma",
    "mixtral": "mistral",
    "phi": "phi2",
    "qwen2": "qwen2",
    "dbrx": "chatml",
    "cohere": "cohere",
}

class Template(object):
    def __init__(self,
                 template: str = "chatml",
                 exception_callback = None
                 ):
        loader = jinja2.PackageLoader('sillm', 'templates')
        env = jinja2.Environment(loader=loader, trim_blocks=True, lstrip_blocks=True)

        if exception_callback is None:
            env.globals["raise_exception"] = self._raise_exception
        else:
            env.globals["raise_exception"] = exception_callback

        fname_template = template + ".jinja"
        self.template = env.get_template(fname_template)

        logger.info(f"Initialized conversation template: {template}")

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
                            add_generation_prompt: bool = False
                            ):
        return self.template.render(messages=messages, add_generation_prompt=add_generation_prompt)
    
class AutoTemplate(Template):
    def __init__(self,
                 tokenizer
                 ):
        self.tokenizer = tokenizer

        logger.info(f"Initialized built-in conversation template from tokenizer")
    
    def apply_chat_template(self,
                            messages: list,
                            add_generation_prompt: bool = False
                            ):
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)