"""Homebrew by Empero: brew your own language model.

The package is split in two halves that must stay independent:

* the *control plane* (agent, UI, backends, data prep, packaging) runs on the
  user's laptop and never imports torch;
* the *worker* (``homebrew_ai.train``) runs wherever the GPU is and only needs
  torch/transformers/peft plus a few light dependencies.

Keep this module free of imports so both halves can load it cheaply.
"""

__version__ = "0.1.0"
