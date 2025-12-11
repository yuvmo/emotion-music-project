from config.settings import N_RETRY, RETRY_DELAY, TASK_SEMAPHORE

import asyncio
import logging
from pydantic import PrivateAttr
from langchain_gigachat import GigaChat as LangGigaChat

from config.settings import (
    GIGACHAT_API_KEY,
    DEFAULT_MODEL,
)


logger = logging.getLogger(__name__)


class LangChainGigaChatWithLimit(LangGigaChat):
    _semaphore: asyncio.Semaphore = PrivateAttr()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._semaphore = TASK_SEMAPHORE

    async def _agenerate(self, *args, **kwargs):
        last_exception = None

        for attempt in range(N_RETRY):
            try:
                async with self._semaphore as acquired_request_id:
                    logger.debug(f"Semaphore acquired for request {acquired_request_id}, {self._semaphore}")
                    result = await super()._agenerate(*args, **kwargs)
                logger.debug(f"Semaphore released for request {acquired_request_id}")
                return result
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1}/{N_RETRY} failed: {e}")

                if attempt < N_RETRY - 1:
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    logger.exception(f"All {N_RETRY} attempts failed")

        raise last_exception


def get_llm() -> LangChainGigaChatWithLimit:
    if not GIGACHAT_API_KEY:
        raise ValueError("Не найден токен GIGACHAT_API_KEY в .env файле")

    return LangChainGigaChatWithLimit(
        credentials=GIGACHAT_API_KEY,
        verify_ssl_certs=False,
        scope="GIGACHAT_API_CORP",
        timeout=600,
        temperature=0.2,
        model=DEFAULT_MODEL
    )