from __future__ import annotations

import torch

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.engine import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


@EngineRegistry.register(model_type="kd_language_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class KDFSDPEngineWithLMHead(FSDPEngineWithLMHead):
    """FSDP LM-head engine variant that keeps raw logits available to KD loss."""

    def prepare_model_outputs(self, output, output_args, micro_batch):
        kd_logits = self._extract_kd_logits(output, output_args=output_args, micro_batch=micro_batch)
        model_output = super().prepare_model_outputs(output=output, output_args=output_args, micro_batch=micro_batch)
        model_output["logits"] = kd_logits
        return model_output

    def _extract_kd_logits(self, output, *, output_args, micro_batch):
        logits = output.logits
        input_ids = micro_batch["input_ids"]
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)

        if use_remove_padding:
            logits = logits.squeeze(0)
            if self.use_ulysses_sp:
                from verl.utils.ulysses import gather_outputs_and_unpad

                logits = gather_outputs_and_unpad(
                    logits,
                    gather_dim=0,
                    unpad_dim=0,
                    padding_size=output_args["pad_size"],
                )
            if pad_mode == DatasetPadMode.NO_PADDING:
                return torch.nested.nested_tensor_from_jagged(logits, input_ids.offsets())
            raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        if pad_mode == DatasetPadMode.NO_PADDING and getattr(input_ids, "is_nested", False):
            cu_seqlens = input_ids.offsets()
            seq_lengths = cu_seqlens.diff()
            starts = torch.zeros_like(seq_lengths, dtype=torch.int64)
            logits = torch.nested.narrow(logits, 1, starts, seq_lengths, layout=torch.jagged)
            logits_rmpad = torch.cat([tensor for tensor in logits.unbind()])
            return torch.nested.nested_tensor_from_jagged(logits_rmpad, cu_seqlens)
        return logits
