import pathlib

from typing import List

import sillm.args

class Tokenizer():
    def encode(self, s: str, eos: bool = False) -> List[int]:
        raise NotImplementedError("Class tokenizer.Tokenizer is used for inheritance only")
    
    def decode(self, t: List[int]) -> str:
        raise NotImplementedError("Class tokenizer.Tokenizer is used for inheritance only")
    
    @property
    def vocab_size(self) -> int:
        raise NotImplementedError("Class tokenizer.Tokenizer is used for inheritance only")
    
########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/e74889d0fa0fb49d95bfdf6a1dcad907713eb50e/llms/mixtral/mixtral.py#L221
########
class SentencePieceTokenizer(Tokenizer):
    """
    Tokenizer wrapper for SentencePiece.
    """
    def __init__(self,
                 tokenizer_path: str,
                 args: sillm.args.ModelArgs
                 ):
        """
        Args:
            tokenizer_path: Path to tokenizer file.
        """
        assert pathlib.Path(tokenizer_path).exists(), tokenizer_path

        try:
            import sentencepiece
        except ImportError:
            raise ImportError("Please install sentencepiece library to use SentencePieceTokenizer")

        self._model = sentencepiece.SentencePieceProcessor(model_file=tokenizer_path)
        self._sep = "▁"

        assert self._model.vocab_size() == self._model.get_piece_size()

        if args.bos_token_id is None:
            self.bos_id = self._model.bos_id()
        else:
            self.bos_id = args.bos_token_id
        if args.eos_token_id is None:
            self.eos_id = self._model.eos_id()
        else:
            self.eos_id = args.eos_token_id

    def encode(self, s: str, eos: bool = False) -> List[int]:
        """
        Encode string.
        Args:
            s: String to encode.
            eos: Whether to append EOS token.
        Returns:
            Encoded tokens.
        """
        tokens = [self.bos_id, *self._model.encode(s)]
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
    def vocab_size(self) -> int:
        """
        Vocabulary size.
        """
        return self._model.vocab_size()
    
class TransformerTokenizer(Tokenizer):
    """
    Tokenizer wrapper for Transformers AutoTokenizer.
    """
    def __init__(self,
                 tokenizer_dir: str,
                 args: sillm.args.ModelArgs
                 ):
        """
        Args:
            tokenizer_path: Path to tokenizer directory.
        """
        assert pathlib.Path(tokenizer_dir).exists(), tokenizer_dir

        try:
            import transformers
        except ImportError:
            raise ImportError("Please install transformers library to use TransformerTokenizer")

        self._model = transformers.AutoTokenizer.from_pretrained(tokenizer_dir)

        if args.bos_token_id is None:
            self.bos_id = self._model.bos_token_id
        else:
            self.bos_id = args.bos_token_id
        if args.eos_token_id is None:
            self.eos_id = self._model.eos_token_id
        else:
            self.eos_id = args.eos_token_id

    def encode(self, s: str, eos: bool = False) -> List[int]:
        """
        Encode string.
        Args:
            s: String to encode.
            eos: Whether to append EOS token.
        Returns:
            Encoded tokens.
        """
        tokens = self._model.encode(s, add_special_tokens=False)
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
        return self._model.decode(t)

    @property
    def vocab_size(self) -> int:
        """
        Vocabulary size.
        """
        return self._model.vocab_size