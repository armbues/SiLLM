import pathlib

from typing import List

import sentencepiece

########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/e74889d0fa0fb49d95bfdf6a1dcad907713eb50e/llms/mixtral/mixtral.py#L221
########
class Tokenizer:
    """
    Tokenizer wrapper.
    """
    def __init__(self, tokenizer_path: str):
        """
        Args:
            tokenizer_path: Path to tokenizer file.
        """
        assert pathlib.Path(tokenizer_path).exists(), tokenizer_path

        self._path = tokenizer_path
        self._model = sentencepiece.SentencePieceProcessor(model_file=tokenizer_path)
        self._sep = "▁"

        assert self._model.vocab_size() == self._model.get_piece_size()

    def encode(self, s: str, eos: bool = False) -> List[int]:
        """
        Encode string.
        Args:
            s: String to encode.
            eos: Whether to append EOS token.
        Returns:
            Encoded tokens.
        """
        tokens = [self._model.bos_id(), *self._model.encode(s)]
        if eos:
            tokens.append(self.eos_id)

        return tokens
    
    def decode(self, t: List[int]) -> str:
        """
        Decode tokens.
        Args:
            t: Tokens to decode.
        Returns:
            Decoded string.
        """
        s = self._model.decode(t)
        if t and self._model.id_to_piece(t[0])[0] == self._sep:
            return " " + s
        
        return s

    @property
    def eos_id(self) -> int:
        """
        EOS token ID.
        """
        return self._model.eos_id()
    
    @property
    def pad_id(self) -> int:
        """
        PAD token ID.
        """
        return self._model.pad_id()

    @property
    def vocab_size(self) -> int:
        """
        Vocabulary size.
        """
        return self._model.vocab_size()