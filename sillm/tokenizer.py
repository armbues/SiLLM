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

    def encode(self,
               s: str,
               bos: bool = True,
               eos: bool = False) -> List[int]:
        """
        Encode string.
        Args:
            s: String to encode.
            eos: Whether to append EOS token.
        Returns:
            Encoded tokens.
        """
        if bos is True:
            tokens = [self.bos_id, *self._model.encode(s)]
        else:
            tokens = self._model.encode(s)
            
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

########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/854ad8747a9c703773adf8866602b114f68aa54a/llms/gguf_llm/utils.py#L5
########
class GGUFTokenizer(SentencePieceTokenizer):
    """
    Tokenizer wrapper for GGUF.
    """
    def __init__(self,
                 metadata: dict
                 ):
        """
        Args:
            tokenizer_path: Path to tokenizer directory.
        """
        try:
            import sentencepiece
            import sentencepiece.sentencepiece_model_pb2
        except ImportError:
            raise ImportError("Please install sentencepiece library to use SentencePieceTokenizer")
        
        tokens = metadata["tokenizer.ggml.tokens"]

        self.unk_id = metadata["tokenizer.ggml.unknown_token_id"].item() if "tokenizer.ggml.unknown_token_id" in metadata else 0
        self.bos_id = metadata["tokenizer.ggml.bos_token_id"].item()
        self.eos_id = metadata["tokenizer.ggml.eos_token_id"].item()
        self._sep = "▁"

        normalizer_spec = sentencepiece.sentencepiece_model_pb2.NormalizerSpec(
            name="identity",
            precompiled_charsmap=b"",
            add_dummy_prefix=True,
            remove_extra_whitespaces=False,
            normalization_rule_tsv=b"",
        )
        trainer_spec = sentencepiece.sentencepiece_model_pb2.TrainerSpec(
            model_type="BPE",
            vocab_size=len(tokens),
            input_format="text",
            split_by_unicode_script=True,
            split_by_whitespace=True,
            split_by_number=True,
            treat_whitespace_as_suffix=False,
            split_digits=True,
            allow_whitespace_only_pieces=True,
            vocabulary_output_piece_score=True,
            byte_fallback=True,
            unk_id=self.unk_id,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            pad_id=-1,
            unk_piece=tokens[self.unk_id],
            bos_piece=tokens[self.bos_id],
            eos_piece=tokens[self.eos_id],
            pad_piece="<pad>",
            pretokenization_delimiter="",
        )
        model_proto = sentencepiece.sentencepiece_model_pb2.ModelProto(trainer_spec=trainer_spec, normalizer_spec=normalizer_spec)
        scores = metadata.get("tokenizer.ggml.scores", None)
        scores = scores.tolist() if scores is not None else None
        token_types = metadata.get("tokenizer.ggml.token_type", None)
        token_types = token_types.tolist() if token_types is not None else None

        for i, token in enumerate(tokens):
            score = scores[i] if scores else 0
            token_type = token_types[i] if token_types else 0
            model_proto.pieces.append(sentencepiece.sentencepiece_model_pb2.ModelProto.SentencePiece(piece=token, score=score, type=token_type))
        
        self._model = sentencepiece.SentencePieceProcessor(model_proto=model_proto.SerializeToString())