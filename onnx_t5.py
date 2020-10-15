import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from psutil import cpu_count
from tqdm import trange
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, Seq2SeqLMOutput, Seq2SeqModelOutput

# Constants from the performance optimization available in onnxruntime
# It needs to be done before importing onnxruntime
os.environ["OMP_NUM_THREADS"] = str(cpu_count(logical=True))
os.environ["OMP_WAIT_POLICY"] = "ACTIVE"

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers

from onnx_utils import generate_onnx_representation

ONNX_CACHE_DIR = Path(os.path.dirname(__file__)).parent.joinpath(".onnx")
logger = logging.getLogger(__name__)


def create_model_for_provider(model_path: str, provider: str) -> InferenceSession:

    assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"

    # Few properties that might have an impact on performances (provided by MS)
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    # Load the model as a graph and prepare the CPU backend
    session = InferenceSession(model_path, options, providers=[provider])
    session.disable_fallback()

    return session


class T5Encoder(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]


class T5Decoder(torch.nn.Module):
    def __init__(self, decoder, lm_head, config):
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.config = config

    def forward(self, input_ids, encoder_hidden_states):
        decoder_output = self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states)[0] * (
            self.config.d_model ** -0.5
        )
        return self.lm_head(decoder_output)


class OnnxT5(GenerationMixin):
    def __init__(self, model_name_or_path, onnx_path, config=None):

        self.model_name_or_path = Path(model_name_or_path)
        self.onnx_path = Path(onnx_path)
        self.mode_base_name = self.model_name_or_path.stem
        self.encoder_path = self.onnx_path.joinpath(f"{self.mode_base_name}_encoder.onnx")
        self.decoder_path = self.onnx_path.joinpath(f"{self.mode_base_name}_decoder.onnx")

        if not (self.encoder_path.exists and self.decoder_path.exists):
            self._export_onnx_graph()

        self.encoder_sess = create_model_for_provider(self.encoder_path.as_posix(), "CPUExecutionProvider")
        self.decoder_sess = create_model_for_provider(self.decoder_path.as_posix(), "CPUExecutionProvider")
        self.config = config

    def __call__(self, input_ids=None, attention_mask=None, decoder_input_ids=None, encoder_outputs=None, **kwargs):
        if input_ids is not None:
            return self._encoder_forward(input_ids=input_ids, attention_mask=attention_mask)

        inputs = {
            "input_ids": decoder_input_ids.cpu().detach().numpy(),
            "encoder_outputs": encoder_outputs.cpu().detach().numpy(),
        }
        lm_logits = self.decoder_sess.run(None, **inputs)
        lm_logits = torch.from_numpy(lm_logits)
        return Seq2SeqLMOutput(logits=lm_logits)

    def _encoder_forward(self, input_ids=None, attention_mask=None):
        inputs = {
            "input_ids": input_ids.cpu().detach().numpy(),
            "attention_mask": attention_mask.cpu().detach().numpy(),
        }
        last_hidden_state = self.encoder_sess.run(None, **inputs)
        last_hidden_state = torch.from_numpy(last_hidden_state)
        return BaseModelOutputWithPast(last_hidden_state=last_hidden_state)

    def get_encoder(self):
        return self

    def _export_onnx_graph(self):
        self.onnx_path.mkdir(parents=True, exist_ok=True)
        generate_onnx_representation(
            self.model_name_or_path, self.encoder_path.as_posix(), self.decoder_path.as_posix()
        )
