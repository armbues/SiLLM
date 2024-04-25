__version__ = "0.1.3"

from sillm.core.loader import load
from sillm.training.dataset import load_dataset
from sillm.core.template import init_template

from sillm.core.llm import LLM
from sillm.core.conversation import Conversation, AutoConversation
from sillm.core.template import Template, AutoTemplate
from sillm.training.lora import TrainableLoRA
from sillm.training.dpo import TrainableDPO
from sillm.training.dataset import DatasetCompletion, DatasetInstruct, DatasetDPO
from sillm.models.args import ModelArgs