## Experiment Settings

- Learning rate
    - PLM: 2e-5
    - Other LM: 1e-3

- Weight decay
    - Resume: 0.01
    - Others: 0.001

- Batch size
    - K <= 1000: 4
    - K > 1000: 8

- synthetic_weights
    - K <= 500: 0.4 (Resume) or 1
    - K > 500: 0.15 (Resume) or 1 (Others)

- decay_weights
    - K < 500: 0
    - K >= 500: 0.13
