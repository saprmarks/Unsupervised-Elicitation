import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Optional, Union

import attrs
import nnsight
from nnsight import LanguageModel
from termcolor import cprint

from core.llm_api.base_llm import PRINT_COLORS, LLMResponse, ModelAPIProtocol
from core.llm_api.openai_llm import OAIChatPrompt

NNSIGHT_MODELS = {
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-70B",
    "meta-llama/Llama-3.1-405B",
}

LOGGER = logging.getLogger(__name__)

@attrs.define()
class NNSightModel(ModelAPIProtocol):
    api_key: str
    print_prompt_and_response: bool = False
    model: Optional[LanguageModel] = attrs.field(init=False, default=None)

    def __attrs_post_init__(self):
        nnsight.api_key = self.api_key

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
        response = None
        duration = None

        # Initialize model if not already done
        if self.model is None:
            self.model = LanguageModel(model_id)

        # Convert prompt to string if it's a chat prompt
        if isinstance(prompt, list):
            prompt_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in prompt])
        else:
            prompt_str = prompt

        for i in range(max_attempts):
            try:
                api_start = time.time()
                with self.model.generate(prompt_str, max_new_tokens=kwargs.get("max_tokens", 2000)) as generator:
                    completion = ""
                    logprobs = []
                    for token in generator:
                        completion += token
                        if hasattr(token, "logprobs"):
                            logprobs.append(token.logprobs)
                api_duration = time.time() - api_start
                response = {
                    "text": completion,
                    "logprobs": logprobs if logprobs else None
                }
            except Exception as e:
                error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}"
                LOGGER.warn(f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i})")
                await asyncio.sleep(1.5**i)
            else:
                break

        if response is None:
            raise RuntimeError(f"Failed to get a response from the API after {max_attempts} attempts.")

        duration = time.time() - start
        LOGGER.debug(f"Completed call to {model_id} in {duration}s")

        llm_response = LLMResponse(
            model_id=model_id,
            completion=response["text"],
            stop_reason="stop",  # NNSight doesn't provide stop reason
            duration=duration,
            api_duration=api_duration,
            cost=0,  # NNSight doesn't track costs
            logprobs=response["logprobs"]
        )

        if self.print_prompt_and_response or print_prompt_and_response:
            cprint("Prompt:", "white")
            cprint(prompt_str, PRINT_COLORS["user"])
            cprint(f"Response ({llm_response.model_id}):", "white")
            cprint(f"{llm_response.completion}", PRINT_COLORS["assistant"], attrs=["bold"])
            print()

        return [{"prompt": prompt, "response": llm_response.to_dict()}] 