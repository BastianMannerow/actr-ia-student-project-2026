# Runtime and State-Graph correction

## Failure reproduced from simulation_history(3)

The exported run contained 28,709 events but no Manual-module activity. Both Example agents entered `runtimeerror_floor_landmarks_not_visible` immediately after `P01_identify_self`. The run then alternated only between `P03_request_situational_assessment` and `P08_adapter_error_recover`, so no production with a `+manual>` request could become eligible.

## Correction

- `Example` declares perfect line of sight as required by the task.
- `AgentConstruct` converts that declaration into a complete pyactr visual frame even when an older GUI configuration still stores LOS 32.
- `ExampleAdapter` continues to read only `pyactr_extension.visual_stimuli()` and now reconstructs platform left/center/right samples correctly.
- The State Graph retains the adapter as a separate dashed transition layer while listing only official ACT-R modules and declared buffers. It is generated only after opening Agent Analysis and explicitly selecting an agent.
- Windows receives the multi-resolution ICO through Qt, `WM_SETICON`, and the native window-class `HICON`/`HICONSM` fields.

## Verification

- The complete automated regression suite passes.
- A run with saved LOS 32 produces `manual / COMMAND: press_key` events and physical displacement.
- The generated `docs/analysis/Example_state.json` contains no synthetic protocol module, includes Manual-module transitions, and contains no autonomous reset production.
