from typing import Tuple

import litserve as ls
import torch
from litserve.specs.openai import ChatCompletionRequest
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

from src.utils import parse_messages


class MiniCPMVLitAPI(ls.LitAPI):
    def setup(self, device):
        model_id = "openbmb/MiniCPM-V-2_6-int4"

        # specify how to quantize the model
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            # quantization_config=quantization_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        # set model to eval mode
        self.model.eval().to(device)

    def decode_request(self, request: ChatCompletionRequest, context):
        context["params"] = {
            "temperature": request.temperature or 0.7,
            "max_new_tokens": request.max_tokens if request.max_tokens else 1024,
            "top_p": request.top_p or 0.8,
            "top_k": 100,
            "repetition_penalty": 1.05,
        }

        # parse messages
        system_prompt, messages = parse_messages(request)
        return system_prompt, messages

    def predict(self, inputs: Tuple, context):
        system_prompt, messages = inputs
        res = self.model.chat(
            image=None,
            msgs=messages,
            tokenizer=self.tokenizer,
            sampling=True,
            temperature=context["temperature"],
            stream=True,
            system_prompt=system_prompt,
            **context["params"],
        )

        for text in res:
            yield text


if __name__ == "__main__":
    api = MiniCPMVLitAPI()
    server = ls.LitServer(api, spec=ls.OpenAISpec())
    server.run(port=8000)
