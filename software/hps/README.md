# HPS software (ARM Cortex-A9, rsYocto Linux)

Copy these files to `/root/ai` on the board.

| File | Role |
|---|---|
| `bottle_ai_live_robuste.sh` | Called by the PC for each image; runs the inference program |
| *(C program)* | Writes the image to `input_ram`, starts Conv1 → Conv2 → Conv3, reads `pool3_ram`, computes MaxPool3 + GAP + dense layers on the ARM |

The program must print one line in this format, which the PC client parses:

```
RESULT class=<0|1> fpga_ms=<float> arm_ms=<float> total_ms=<float>
```
