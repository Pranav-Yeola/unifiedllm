class UnifiedLLMError(Exception):
    """Base exception for unifiedllm."""

    pass


class ProviderNotSupportedError(UnifiedLLMError):
    pass


class MissingAPIKeyError(UnifiedLLMError):
    pass


class ProviderRequestError(UnifiedLLMError):
    pass
