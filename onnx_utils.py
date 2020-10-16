import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from onnx_t5 import T5Decoder, T5Encoder


def create_t5_encoder_decoder(model="t5-base"):
    """Generates an encoder and a decoder model with a language model head from a pretrained huggingface model
    Args:
        model (str): Name of a pretrained model, or path to a pretrained / finetuned version of T5
    Returns:
        t5_encoder: pytorch t5 encoder with a wrapper to output only the hidden states
        t5_decoder: pytorch t5 decoder with a language modeling head
    """

    # T5 is an encoder / decoder model with a language modeling head on top.
    # We need to separate those out for efficient language generation
    if isinstance(model, str):
        model = T5ForConditionalGeneration.from_pretrained(model)

    encoder = model.encoder
    decoder = model.decoder
    lm_head = model.lm_head

    t5_encoder = T5Encoder(encoder)
    t5_decoder = T5Decoder(decoder, lm_head, model.config)
    return t5_encoder, t5_decoder


def generate_onnx_representation(model, encoder_path, decoder_path):
    """Exports a given huggingface pretrained model, or a given model and tokenizer, to onnx
    Args:
        pretrained_version (str): Name of a pretrained model, or path to a pretrained / finetuned version of T5
        output_prefix (str): Path to the onnx file
    """

    simplified_encoder, decoder_with_lm_head = create_t5_encoder_decoder(model)

    # Example sequence
    input_ids = torch.tensor([[42] * 512])
    attention_mask = torch.ones((1, 512))

    # Exports to ONNX
    _ = torch.onnx._export(
        simplified_encoder,
        (input_ids, attention_mask),
        encoder_path,
        export_params=True,
        opset_version=12,
        input_names=["input_ids", "attention_mask"],
        output_names=["encoder_hidden_states"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "encoder_hidden_states": {0: "batch", 1: "sequence"},
        },
    )

    _ = torch.onnx.export(
        decoder_with_lm_head,
        (input_ids, simplified_encoder(input_ids)),
        decoder_path,
        export_params=True,
        opset_version=12,
        input_names=["input_ids", "encoder_hidden_states"],
        output_names=["lm_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "encoder_hidden_states": {0: "batch", 1: "sequence"},
            "lm_logits": {0: "batch", 1: "sequence"},
        },
    )
