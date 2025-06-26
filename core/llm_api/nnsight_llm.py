import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Optional, Union

import attrs
from vllm import LLM, SamplingParams
# import nnsight
# from nnsight.modeling.vllm import VLLM
from termcolor import cprint
import torch

from core.llm_api.base_llm import PRINT_COLORS, LLMResponse, ModelAPIProtocol
from core.llm_api.openai_llm import OAIChatPrompt

NNSIGHT_MODELS = {
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Llama-3.1-70B",
    "meta-llama/Llama-3.1-405B",
}

LOGGER = logging.getLogger(__name__)

# Global counter to track model initializations
_model_initialization_count = 0

@attrs.define()
class NNSightModel(ModelAPIProtocol):
    # api_key: str
    print_prompt_and_response: bool = False
    model: Optional[LLM] = attrs.field(init=False, default=None)
    model_lock: asyncio.Lock = attrs.field(init=False, default=attrs.Factory(asyncio.Lock))

    # def __attrs_post_init__(self):
    #     nnsight.api_key = self.api_key

    @staticmethod
    def _create_prompt_history_file(prompt):
        filename = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}_prompt.txt"
        with open(os.path.join("prompt_history", filename), "w") as f:
            json_str = json.dumps(prompt, indent=4)
            json_str = json_str.replace("\\n", "\n")
            f.write(json_str)
        return filename

    @staticmethod
    def _add_response_to_prompt_file(prompt_file, response):
        with open(os.path.join("prompt_history", prompt_file), "a") as f:
            f.write("\n\n======RESPONSE======\n\n")
            json_str = json.dumps(response.to_dict(), indent=4)
            json_str = json_str.replace("\\n", "\n")
            f.write(json_str)

    @torch.no_grad()
    async def __call__(
        self,
        model_ids: list[str],
        prompt: Union[str, OAIChatPrompt],
        print_prompt_and_response: bool,
        max_attempts: int,
        **kwargs,
    ) -> list[LLMResponse]:
        start = time.time()
        assert len(model_ids) == 1, "NNSight implementation only supports one model at a time."
        model_id = model_ids[0]
        assert model_id in NNSIGHT_MODELS, f"Invalid model id: {model_id}"
        
        LOGGER.debug(f"Making {model_id} call")

        # Convert prompt to string if it's a chat prompt
        if isinstance(prompt, list):
            prompt_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in prompt])
        else:
            prompt_str = prompt


        sampling_params = SamplingParams(
            **{
                key: value
                for key, value in kwargs.items()
                if key not in ("save_path", "metadata")
            }
        )
        
        api_start = time.time()
        
        # Use a single model instance with lock for thread safety
        async with self.model_lock:
            # Initialize model if not already done (inside lock to prevent race condition)
            if self.model is None:
                global _model_initialization_count
                _model_initialization_count += 1
                LOGGER.info(f"Initializing NNSight model #{_model_initialization_count}: {model_id}")
                self.model = LLM(model_id, device="auto")
                LOGGER.info(f"Model initialization #{_model_initialization_count} complete: {model_id}")
            else:
                LOGGER.debug(f"Reusing existing NNSight model: {model_id}")
            
            for i in range(max_attempts):
                try:
                    outputs = self.model.generate(prompt_str, sampling_params=sampling_params)

                    responses = []
                    for output in outputs[0].outputs:

                        if kwargs.get("logprobs", None):
                            logprobs = [
                                {
                                    logprob.decoded_token : logprob.logprob
                                    for logprob in token_logprobs.values()
                                }
                                for token_logprobs in output.logprobs
                            ]
                        else:
                            logprobs = None
                        
                        responses.append(
                            {
                                "text": output.text,
                                "logprobs": logprobs
                            }
                        )
                    break

                except Exception as e:
                    error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}"
                    LOGGER.warn(f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i})")
                    await asyncio.sleep(1.5**i)
            else:
                raise RuntimeError(f"Failed to get a response from the API after {max_attempts} attempts.")
        
        api_duration = time.time() - api_start
        duration = time.time() - start
        LOGGER.debug(f"Completed call to {model_id} in {duration}s")

        llm_responses = [
            LLMResponse(
                model_id=model_id,
                completion=response["text"],
                stop_reason="stop",  # NNSight doesn't provide stop reason
                duration=duration,
                api_duration=api_duration,
                cost=0,  # NNSight doesn't track costs
                logprobs=response["logprobs"]
            )
            for response in responses
        ]

        if self.print_prompt_and_response or print_prompt_and_response:
            cprint("Prompt:", "white")
            cprint(prompt_str, PRINT_COLORS["user"])
            for response in llm_responses:
                cprint(f"Response ({response.model_id}):", "white")
                cprint(f"{response.completion}", PRINT_COLORS["assistant"], attrs=["bold"])
            print()

        return [{"prompt": prompt, "response": response.to_dict()} for response in llm_responses] 