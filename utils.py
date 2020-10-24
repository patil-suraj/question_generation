# adapted from https://github.com/huggingface/transformers/blob/master/examples/seq2seq/utils.py

from typing import Callable, Dict, Iterable, List

import torch
from torch import nn

# these functions are taken from transformers repo


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())


def freeze_params(model: nn.Module):
    for par in model.parameters():
        par.requires_grad = False


def freeze_embeds(model: nn.Module):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    try:
        freeze_params(model.model.shared)
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)
    except AttributeError:
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)


def assert_not_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    npars = len(model_grads)
    assert any(model_grads), f"none of {npars} weights require grad"


def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


# prepares lm_labels from target_ids, returns examples with keys as expected by the forward method
# this is necessacry because the trainer directly passes this dict as arguments to the model
# so make sure the keys match the parameter names of the forward method
class T2TDataCollator:
    def __init__(self, tokenizer, model_type="t5", mode="training", using_tpu=False):
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.mode = mode
        self.using_tpu = using_tpu

    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([example["source_ids"] for example in batch])
        target_ids = torch.stack([example["target_ids"] for example in batch])
        attention_mask = torch.stack([example["attention_mask"] for example in batch])

        pad_token_id = self.tokenizer.pad_token_id

        # don't trim on tpu, for some reason trimming leads to slower training on TPU
        if not self.using_tpu:
            input_ids, attention_mask = trim_batch(input_ids, pad_token_id, attention_mask=attention_mask)
            target_ids = trim_batch(target_ids, pad_token_id)

        if self.model_type == "t5":
            lm_labels = target_ids.clone()
            decoder_input_ids = self._shift_right_t5(lm_labels)
            if self.mode == "training":
                lm_labels[lm_labels[:, :] == pad_token_id] = -100
        else:
            decoder_input_ids = target_ids[:, :-1].contiguous()
            lm_labels = target_ids[:, 1:].clone()
            if self.mode == "training":
                lm_labels[target_ids[:, 1:] == pad_token_id] = -100

        params = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": lm_labels,
            "decoder_input_ids": decoder_input_ids,
        }

        return params

    def _shift_right_t5(self, input_ids):
        decoder_start_token_id = self.tokenizer.pad_token_id
        pad_token_id = self.tokenizer.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `labels` has only positive values and -100"

        return shifted_input_ids
