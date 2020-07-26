from typing import Any, Dict, Union

import torch
from torch import nn

from transformers import Trainer as HFTrainer
from transformers.file_utils import is_apex_available

if is_apex_available():
    from apex import amp

from utils import label_smoothed_nll_loss

class Trainer(HFTrainer):
    def __init__(self, label_smoothing: float = 0, **kwargs):
        super().__init__(**kwargs)
        self.label_smoothing = label_smoothing
    
    # override to support label smoothing
    def _training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], optimizer: torch.optim.Optimizer
    ) -> float:
        model.train()
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)


        # Our model outputs do not work with DataParallel, so forcing return tuple.
        if isinstance(model, nn.DataParallel):
            inputs["return_tuple"] = True

        if self.label_smoothing == 0:
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
        else:
            labels = inputs.pop("labels")
            labels[labels == -100] = model.config.pad_token_id
            outputs = model(**inputs)
            lprobs = torch.nn.functional.log_softmax(outputs[0], dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, labels, self.label_smoothing, ignore_index=model.config.pad_token_id
            )

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.item()
