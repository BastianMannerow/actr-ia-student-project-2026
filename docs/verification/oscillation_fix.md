# Oscillation fix verification

## Reproduced failure

Source history: `simulation_history(4).zip`, seed `612592664`, scenario `paired_left`.

- The circle repeatedly approached the priority ledge, jumped right, landed on the same upper-floor support, and returned left.
- The prior progress model kept `stuck=no` and `no_progress_cycles=0`, because displacement itself was counted as progress.
- The original 2.4-unit staging tolerance released the jump before the body had cleared the low-ceiling collision envelope.
- The supporting rectangle accumulated waiting cycles although partner movement was rational joint progress.

## Corrections

- top and bottom visual samples for every platform;
- exact adapter reconstruction of platform top, bottom, width, and height;
- collision-aware take-off point outside overlapping low ceilings;
- Manual-priming clearance and narrow take-off tolerance;
- route-subgoal distance and best-distance memory;
- support-surface outcome memory and failed-jump episodes;
- persistent stuck diagnosis until observed progress;
- recovery direction committed toward the extended route subgoal;
- partner movement counted as progress during support/monitoring phases.

## Deterministic replay

The replay starts the circle at `x=55.5` on the upper floor, keeps the rectangle at `x=8.5`, and preserves the original priority geometry. The circle reaches approximately `x=44.19`, primes and executes one rightward jump, clears the low ceiling, lands on the priority ledge, and collects the diamond in 2,394 or fewer cognitive steps. No runtime error and no left/right recovery loop occur.

## Automated validation

- 23 regression tests pass.
- The exact platform collision envelope is asserted against physics objects.
- The uploaded seed and terminal geometry are replayed in a dedicated integration test.
- State-graph analysis reports no unreachable productions or dead-end states.
