"""Fail-fast architecture/revision/tokenizer compatibility audit."""

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer

from stitching_core import BASE, GUARD, REVISIONS


def main():
    print(f"torch={torch.__version__} transformers={transformers.__version__}")
    if transformers.__version__ != "5.16.1":
        raise RuntimeError("This repository is pinned and tested with transformers==5.16.1")
    configs = {name: AutoConfig.from_pretrained(name, revision=REVISIONS[name])
               for name in (BASE, GUARD)}
    tokenizers = {name: AutoTokenizer.from_pretrained(name, revision=REVISIONS[name])
                  for name in (BASE, GUARD)}
    for name in (BASE, GUARD):
        config = configs[name]
        print(name, REVISIONS[name], config.architectures, config.hidden_size,
              config.num_hidden_layers, config.num_attention_heads, config.num_key_value_heads)
        if (config.hidden_size, config.num_hidden_layers) != (2560, 36):
            raise RuntimeError(f"Unexpected architecture for {name}")
    left, right = tokenizers[BASE].get_vocab(), tokenizers[GUARD].get_vocab()
    if left != right or tokenizers[BASE].all_special_ids != tokenizers[GUARD].all_special_ids:
        raise RuntimeError("Tokenizer vocabulary or special IDs differ")
    if tokenizers[BASE].chat_template == tokenizers[GUARD].chat_template:
        raise RuntimeError("Expected distinct Base and Guard chat templates")
    print("QWEN INSPECTION: PASS")


if __name__ == "__main__": main()
