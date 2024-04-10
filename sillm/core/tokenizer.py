import pathlib
import shutil

from typing import List

import sillm.models.args as args

class Tokenizer():
    def encode(self,
               s: str,
               bos: bool = True,
               eos: bool = False
               ) -> List[int]:
        raise NotImplementedError("Class tokenizer.Tokenizer is used for inheritance only")
    
    def decode(self,
               t: List[int]
               ) -> str:
        raise NotImplementedError("Class tokenizer.Tokenizer is used for inheritance only")
    
    def has_template(self) -> bool:
        return False
    
    def apply_chat_template(self,
                            *args,
                            **kwargs
                            ):
        raise NotImplementedError("Tokenizer does not support chat templates - check has_template() first")
        
    @property
    def vocab_size(self) -> int:
        raise NotImplementedError("Class tokenizer.Tokenizer is used for inheritance only")
    
    def save(self,
             tokenizer_path: str
             ):
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
                 args: args.ModelArgs
                 ):
        """
        Args:
            tokenizer_path: Path to tokenizer file.
        """
        assert pathlib.Path(tokenizer_path).exists(), tokenizer_path
        self._source = tokenizer_path

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
        self.pad_id = self._model.pad_id()

        self.special_ids = [self.bos_id, self.eos_id, self.pad_id]

    def encode(self,
               s: str,
               bos: bool = True,
               eos: bool = False
               ) -> List[int]:
        """
        Encode string.
        Args:
            s: String to encode.
            bos: Whether to prefix BOS token.
            eos: Whether to append EOS token.
        Returns:
            Encoded tokens.
        """
        if bos:
            tokens = [self.bos_id, *self._model.encode(s)]
        else:
            tokens = self._model.encode(s)

        if eos:
            tokens.append(self.eos_id)

        return tokens
    
    def decode(self,
               t: List[int]
               ) -> str:
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
    
    def save(self,
            tokenizer_path: str
            ):
        tokenizer_path = pathlib.Path(tokenizer_path) / "tokenizer.model"

        shutil.copy(self._source, tokenizer_path)
    
class TransformerTokenizer(Tokenizer):
    """
    Tokenizer wrapper for Transformers AutoTokenizer.
    """
    def __init__(self,
                 tokenizer_dir: str,
                 args: args.ModelArgs
                 ):
        """
        Args:
            tokenizer_path: Path to tokenizer directory.
        """
        assert pathlib.Path(tokenizer_dir).exists(), tokenizer_dir

        try:
            import transformers
            transformers.logging.set_verbosity_error()
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
        if args.pad_token_id is None:
            self.pad_id = self._model.pad_token_id
        else:
            self.pad_id = args.pad_token_id

        self.special_ids = set([self.bos_id, self.eos_id] + self._model.all_special_ids)

    def encode(self,
               s: str,
               bos: bool = True,
               eos: bool = False
               ) -> List[int]:
        """
        Encode string.
        Args:
            s: String to encode.
            bos: Whether to prefix BOS token.
            eos: Whether to append EOS token.
        Returns:
            Encoded tokens.
        """
        if bos:
            tokens = [self.bos_id, *self._model.encode(s, add_special_tokens=False)]
        else:
            tokens = self._model.encode(s, add_special_tokens=False)

        if eos:
            tokens.append(self.eos_id)

        return tokens
    
    def decode(self,
               t: List[int]
               ) -> str:
        """
        Decode tokens.
        Args:
            t: Tokens to decode.
        Returns:
            Decoded string.
        """
        return self._model.decode(t)
    
    def has_chat_template(self) -> bool:
        """
        Check if tokenizer has chat template.
        """
        return hasattr(self._model, "apply_chat_template")
    
    def apply_chat_template(self,
                            *args,
                            **kwargs
                            ):
        """
        Apply chat template.
        """
        return self._model.apply_chat_template(*args, **kwargs)

    @property
    def vocab_size(self) -> int:
        """
        Vocabulary size.
        """
        return self._model.vocab_size
    
    def save(self,
            tokenizer_path: str
            ):
        self._model.save_pretrained(tokenizer_path)

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
            metadata: Model metadata in GGUF format.
        """
        try:
            import sentencepiece
            import sentencepiece.sentencepiece_model_pb2
        except ImportError:
            raise ImportError("Please install sentencepiece and protobuf==3.20.1 libraries to use SentencePieceTokenizer")
        
        tokens = metadata["tokenizer.ggml.tokens"]
        scores = metadata.get("tokenizer.ggml.scores", None)
        scores = scores.tolist() if scores is not None else None
        token_types = metadata.get("tokenizer.ggml.token_type", None)
        byte_fallback = False

        self.bos_id = metadata["tokenizer.ggml.bos_token_id"].item()
        self.eos_id = metadata["tokenizer.ggml.eos_token_id"].item()
        self.unk_id = -1
        if "tokenizer.ggml.unknown_token_id" in metadata:
            self.unk_id = metadata["tokenizer.ggml.unknown_token_id"].item()
        if "tokenizer.ggml.padding_token_id" in metadata:
            self.pad_id = metadata["tokenizer.ggml.padding_token_id"].item()
            pad_token = tokens[self.pad_id]
        else:
            self.pad_id = -1
            pad_token = "<pad>"
        self._sep = "▁"

        self.special_ids = set([self.bos_id, self.eos_id])
        
        if token_types is not None:
            token_types = token_types.tolist()

            # Determine if there enough tokens to use byte fallback
            if token_types.count(6) >= 256:
                # 6 is the token type for BYTE
                byte_fallback = True
            
            # Find UNK token if not provided in metadata
            if self.unk_id < 0:
                for i in range(len(token_types)):
                    if token_types[i] == 2:
                        self.unk_id = i

        # Add UNK token if not defined
        if self.unk_id < 0:
            self.unk_id = len(tokens)
            tokens.append("<unk>")
            if scores:
                scores.append(0)
            if token_types:
                # 2 is the token type for UNKNOWN
                token_types.append(2)

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
            byte_fallback=byte_fallback,
            unk_id=self.unk_id,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            pad_id=self.pad_id,
            unk_piece=tokens[self.unk_id],
            bos_piece=tokens[self.bos_id],
            eos_piece=tokens[self.eos_id],
            pad_piece=pad_token,
            pretokenization_delimiter="",
        )
        normalizer_spec = sentencepiece.sentencepiece_model_pb2.NormalizerSpec(
            name="identity",
            precompiled_charsmap=b"",
            add_dummy_prefix=True,
            remove_extra_whitespaces=False,
            normalization_rule_tsv=b"",
        )
        model_proto = sentencepiece.sentencepiece_model_pb2.ModelProto(trainer_spec=trainer_spec, normalizer_spec=normalizer_spec)

        for i, token in enumerate(tokens):
            score = scores[i] if scores else 0
            token_type = token_types[i] if token_types else 0
            model_proto.pieces.append(sentencepiece.sentencepiece_model_pb2.ModelProto.SentencePiece(piece=token, score=score, type=token_type))
        
        self._model = sentencepiece.SentencePieceProcessor(model_proto=model_proto.SerializeToString())

    def save(self,
            tokenizer_path: str
            ):
        raise NotImplementedError("GGUFTokenizer does not support saving")

class TiktokenTokenizer(Tokenizer):
    """
    Tokenizer wrapper for tiktoken.
    """
    def __init__(self,
                 args: args.ModelArgs,
                 model_name: str = "gpt-4"
                 ):
        try:
            import tiktoken
        except ImportError:
            raise ImportError("Please install tiktoken library to use TiktokenTokenizer")
        
        self._model = tiktoken.encoding_for_model(model_name)

        self.bos_id = args.bos_token_id
        self.eos_id = args.eos_token_id

        self.special_ids = set([self.bos_id, self.eos_id])

    def encode(self,
               s: str,
               bos: bool = True,
               eos: bool = False
               ) -> List[int]:
        """
        Encode string.
        Args:
            s: String to encode.
            bos: Whether to prefix BOS token.
            eos: Whether to append EOS token.
        Returns:
            Encoded tokens.
        """
        tokens = self._model.encode(s)
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens.append(self.eos_id)

        return tokens
    
    def decode(self,
               t: List[int]
               ) -> str:
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
        return self._model.n_vocab
    
    def save(self,
            tokenizer_path: str
            ):
        raise NotImplementedError("TiktokenTokenizer does not support saving")