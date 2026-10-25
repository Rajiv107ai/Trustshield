"""Backward compatibility alias forwarding to core.multimodal_scoring."""

from core.multimodal_scoring import *  # noqa: F401, F403
# Private helpers are excluded from `import *`; re-export explicitly.
from core.multimodal_scoring import _product_fingerprint  # noqa: F401
