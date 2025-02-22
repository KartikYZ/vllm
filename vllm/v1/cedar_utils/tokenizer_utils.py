from typing import Optional
from vllm.transformers_utils.tokenizer_group import BaseTokenizerGroup

_instance_tokenizer: Optional[BaseTokenizerGroup] = None

def set_tokenizer(tokenizer: BaseTokenizerGroup) -> None:
    """Set the global tokenizer instance."""
    global _instance_tokenizer
    _instance_tokenizer = tokenizer

def get_tokenizer() -> Optional[BaseTokenizerGroup]:
    """Get the global tokenizer instance."""
    return _instance_tokenizer 