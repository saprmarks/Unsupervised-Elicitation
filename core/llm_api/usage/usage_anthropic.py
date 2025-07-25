import concurrent.futures
import os
from typing import Optional

import requests

from core.utils import setup_environment


def can_claude_api_take_n_more_concurrents(n: int) -> bool:
    def ping_claude__is_ratelimited() -> Optional[bool]:
        data = {
            "model": "claude-2.0",
            "prompt": "\n\nHuman: Count to 50.\n\nAssistant:",
            "max_tokens_to_sample": 1000,
        }
        headers = {
            "content-type": "application/json",
            "accept": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": f"{os.getenv('ANTHROPIC_API_KEY')}",
        }
        response = requests.post(
            "https://api.anthropic.com/v1/complete",
            headers=headers,
            json=data,
            timeout=20,
        )

        if response.status_code == 200:
            return False
        elif response.status_code == 429:
            return True
        else:
            response.raise_for_status()

    # launch n threads, each of which tries to ping claude
    print(f"Checking if claude can currently take {n} more concurrent requests...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
        futures = []
        for _ in range(n):
            futures.append(executor.submit(ping_claude__is_ratelimited))
        results = [f.result() for f in futures]

    result = not any(results)  # if any result is true, is rate limited
    print(
        f"Claude currently {'can' if result else 'cannot'} take {n} more concurrent requests"
    )
    return result


def binary_search_for_max_concurrent_claude_requests() -> int:
    min_max_new_concurrent_requests = 1
    max_max_new_concurrent_requests = 100
    while (
        min_max_new_concurrent_requests + 5 < max_max_new_concurrent_requests
    ):  # plus five because we don't need it accurate
        mid_test_number = (
            min_max_new_concurrent_requests + max_max_new_concurrent_requests
        ) // 2
        if can_claude_api_take_n_more_concurrents(mid_test_number):
            min_max_new_concurrent_requests = mid_test_number
        else:
            max_max_new_concurrent_requests = mid_test_number
    print(
        f"\n\nFinal result: Claude can currently take roughly {min_max_new_concurrent_requests} more concurrent requests.\n"
    )
    return min_max_new_concurrent_requests


if __name__ == "__main__":
    setup_environment()
    binary_search_for_max_concurrent_claude_requests()
