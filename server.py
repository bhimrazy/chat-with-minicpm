import os
from threading import Thread

import litserve as ls
import torch
from litserve.specs.openai import ChatCompletionRequest
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from utils import parse_messages

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class MiniCPMLitAPI(ls.LitAPI):
    def setup(self, device):
        model_id = "openbmb/MiniCPM-V-2"
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
            torch_dtype="auto",
            quantization_config=quantization_config,
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        # set model to eval mode
        self.model.eval()

    def decode_request(self, request: ChatCompletionRequest, context):
        context["temperature"] = request.temperature or 0.7
        context["max_tokens"] = request.max_tokens if request.max_tokens else 1024
        context["top_p"] = request.top_p

        # parse messages
        messages, images = parse_messages(request)

        model_inputs = {
            "image": images[0] if images else None,
            "messages": messages,
        }
        return model_inputs

    def predict(self, model_inputs: dict, context):
        # streaming
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
        generation_kwargs = dict(
            **model_inputs,
            streamer=streamer,
            context=None,
            sampling=True,
            temperature=context["temperature"],
            max_new_tokens=context["max_tokens"],
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for text in streamer:
            yield text


if __name__ == "__main__":
    api = MiniCPMLitAPI()
    server = ls.LitServer(api, spec=ls.OpenAISpec())
    server.run(port=8000)
