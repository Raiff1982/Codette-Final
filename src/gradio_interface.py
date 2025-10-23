"""
[DEPRECATED] Gradio Interface for Codette
This file is deprecated. Please use the unified interface at /codette_interface.py instead.

Example usage:
    from codette_interface import create_interface
    iface = create_interface("gradio")
    iface.launch()
"""

import warnings
warnings.warn(
    "This interface is deprecated. Please use the unified interface at /codette_interface.py instead.",
    DeprecationWarning,
    stacklevel=2
)

from codette_interface import create_interface

# Provide backwards compatibility
def create_gradio_interface():
    """Backwards compatibility wrapper"""
    warnings.warn(
        "Use create_interface('gradio') instead of create_gradio_interface()",
        DeprecationWarning,
        stacklevel=2
    )
    return create_interface("gradio")