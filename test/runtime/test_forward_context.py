import torch

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata
from tokenspeed.runtime.sampling.logits_layout import LogitsLayoutPlan


def test_logits_metadata_derives_dp_sampling_from_forward_context_layout_plan():
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=1,
        num_extends=0,
        input_num_tokens=1,
        forward_mode=ForwardMode.DECODE,
        logits_layout_plan=LogitsLayoutPlan.dp_all_to_all(
            real_bs=1,
            bucket_bs=4,
            tp_size=4,
            num_tokens_per_req=1,
        ),
    )

    metadata = LogitsMetadata.from_forward_context(
        ctx,
        torch.tensor([1], dtype=torch.int32),
    )

    assert metadata.dp_sampling is True
