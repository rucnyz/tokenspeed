import torch

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata


def test_logits_metadata_uses_forward_context_dp_sampling():
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=1,
        num_extends=0,
        input_num_tokens=1,
        forward_mode=ForwardMode.DECODE,
        dp_sampling=True,
    )

    metadata = LogitsMetadata.from_forward_context(
        ctx,
        torch.tensor([1], dtype=torch.int32),
    )

    assert metadata.dp_sampling is True
