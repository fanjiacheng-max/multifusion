class _C(dict):
    """轻量 ConfigDict，支持 config.key 和 config["key"] 两种访问方式。"""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)
    def __setattr__(self, k, v):
        self[k] = v


def get_IRENE_config():
    config = _C()
    config.hidden_size = 768
    config.token_dim   = 1024          # pre-extracted token dim
    config.n_modalities = 4
    config.mm_layers   = 2             # 前 N 层做 4-modal cross-attn

    transformer = _C()
    transformer.mlp_dim                = 3072
    transformer.num_heads              = 12
    transformer.num_layers             = 12
    transformer.attention_dropout_rate = 0.1
    transformer.dropout_rate           = 0.1
    config.transformer = transformer

    config.modality_dropout_p = 0.1
    config.use_lia = True
    config.lia_temperature = 0.1
    config.lambda_lia = 0.1
    config.lia_exclude_cls = True
    return config


CONFIGS = {'CMR_IRENE': get_IRENE_config()}
