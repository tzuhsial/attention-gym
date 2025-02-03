"""Standard Speculative Streaming Mask"""
from functools import partial
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.attention.flex_attention import _mask_mod_signature


def generate_speculative_streaming_mask_mod(seq_len, is_lossless: bool = False) -> _mask_mod_signature:
    """ Generate a speculative streaming mask mod function. """
    def speculative_streaming_mask_mod(b, h, q_idx, kv_idx):
        q_gram  = q_idx // seq_len
        kv_gram = kv_idx // seq_len
        mod_q_idx = q_idx % seq_len
        mod_kv_idx = kv_idx % seq_len
        # 1. Original LM 
        # this includes causal mask for all streams including original.
        causal_lm = mod_q_idx >= kv_idx
        # 2. Ngram index
        is_ngram = q_gram >= kv_gram
        is_attend = mod_q_idx == mod_kv_idx
        is_ngram_attend = is_ngram & is_attend

        is_spec_streaming = is_ngram_attend | causal_lm

        if is_lossless:
            is_beyond_original = q_gram > 0
            return is_spec_streaming & is_beyond_original

        return is_spec_streaming

    return speculative_streaming_mask_mod


def main(device: str = "cpu"):
    """Visualize the attention scores of causal masking.

    Args:
        device (str): Device to use for computation. Defaults
    """
    import torch
    from attn_gym import visualize_attention_scores

    B, H, SEQ_LEN, HEAD_DIM = 1, 1, 12, 8

    N_GRAM = 3
    def make_tensor(n_gram):
        return torch.ones(B, H, SEQ_LEN * n_gram, HEAD_DIM, device=device)


    query, key = make_tensor(N_GRAM), make_tensor(N_GRAM)

    # block_mask = create_block_mask(causal_mask, B=None, H=None, Q_LEN=SEQ_LEN, KV_LEN=SEQ_LEN)
    speculative_streaming_mask = generate_speculative_streaming_mask_mod(SEQ_LEN, is_lossless=True)

    visualize_attention_scores(query, key, mask_mod=speculative_streaming_mask, device=device, name="speculative_streaming_mask")


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)
