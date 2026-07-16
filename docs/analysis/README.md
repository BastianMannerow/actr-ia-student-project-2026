# Example State Graph

`Example_state.json` is generated from the current `Example` and `ExampleAdapter`.
The graph distinguishes ACT-R production transitions from external adapter
transitions. Adapter edges remain visible as a separate transition layer, while
only official pyactr modules and declared buffers are listed as cognitive module
accesses. No synthetic `protocol` module is used.

The graph is generated lazily in the GUI only after opening **Agent Analysis**
and explicitly selecting an agent or agent type.

Regenerate this file after changing the model or adapter rather than maintaining
graph nodes manually.
