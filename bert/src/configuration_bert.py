from transformers import BertConfig


class BertConfig(BertConfig):

    def __init__(
        self,
        alibi_starting_size: int = 512,
        attention_probs_dropout_prob: float = 0.0,

        # mlp
        use_glu_mlp: bool = True,
        use_monarch_mlp: bool = False,
        monarch_mlp_nblocks: int = 4,

        # position
        use_positional_encodings: bool = False,
        max_position_embeddings: int = 512,

        # architecture selection
        monarch_mixer_sequence_mixing: bool = False,
        residual_long_conv: bool = False,
        
        # hyena and long conv hyperparameters
        bidirectional: bool = True,
        hyena_w_mod: int = 1,
        hyena_filter_dropout: float = 0.2,
        hyena_filter_order: int = 64,
        hyena_training_additions: bool = False,
        
        # efficiency
        use_flash_mm: bool = False,

        # average pooling instead of CLS token
        pool_all: bool = False,

        **kwargs,
    ):
        """Configuration class for MosaicBert.

        Args:
            alibi_starting_size (int): Use `alibi_starting_size` to determine how large of an alibi tensor to
                create when initializing the model. You should be able to ignore this parameter in most cases.
                Defaults to 512.
            attention_probs_dropout_prob (float): By default, turn off attention dropout in Mosaic BERT
                (otherwise, Flash Attention will be off by default). Defaults to 0.0.
        """
        super().__init__(
            attention_probs_dropout_prob=attention_probs_dropout_prob, **kwargs)
        self.alibi_starting_size = alibi_starting_size

        # mlp
        self.use_glu_mlp = use_glu_mlp
        self.use_monarch_mlp = use_monarch_mlp
        self.monarch_mlp_nblocks = monarch_mlp_nblocks

        # positional encodings
        self.use_positional_encodings = use_positional_encodings
        self.max_position_embeddings = max_position_embeddings

        # architecture
        self.monarch_mixer_sequence_mixing = monarch_mixer_sequence_mixing
        self.residual_long_conv = residual_long_conv

        # hyena and long conv hyperparameters
        self.bidirectional = bidirectional
        self.hyena_w_mod = hyena_w_mod
        self.hyena_filter_dropout = hyena_filter_dropout
        self.hyena_filter_order = hyena_filter_order
        self.hyena_training_additions = hyena_training_additions

        # efficiency
        self.use_flash_mm = use_flash_mm

        # average pooling instead of CLS token
        self.pool_all = pool_all
        
