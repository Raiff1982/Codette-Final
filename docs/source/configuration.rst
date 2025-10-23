Configuration
=============

.. _configuration:

Environment Variables
-------------------

The following environment variables can be used to configure Codette:

- ``HUGGINGFACEHUB_API_TOKEN``: HuggingFace API token
- ``OPENAI_API_KEY``: Optional OpenAI API key
- ``LOG_LEVEL``: Logging level
- ``PORT``: Web server port

Model Configuration
-----------------

Codette uses a fallback chain of models:

1. Mistral-7B-Instruct (Primary)
2. Phi-2 (Secondary)
3. GPT-2 (Fallback)

See :class:`src.ai_core.AICore` for implementation details.

Consciousness System
------------------

Memory Management
~~~~~~~~~~~~~~~

- Response memory: Last 50 responses
- Memory context: Last 5 responses for learning
- Memory synthesis: Last 2 responses for consciousness

Quantum States
~~~~~~~~~~~~

Stored in .cocoon files with:

- quantum_state arrays
- chaos_state arrays
- perspective information

See :meth:`src.ai_core.AICore.load_cocoon_data` for details.