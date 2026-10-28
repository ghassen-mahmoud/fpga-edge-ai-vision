# ROM initialization files

Platform Designer loads the weights and biases of each convolution from these files:

| File | ROM | Content |
|---|---|---|
| `conv1_weights_64.mif` | `conv1_weights_rom` | 432 INT8 weights, 8 per 64-bit word |
| `conv1_bias_32.mif` | `conv1_bias_rom` | 16 INT32 biases |
| `conv2_weights_64.mif` | `conv2_weights_rom` | 4 608 INT8 weights, 8 per 64-bit word |
| `conv2_bias_32.mif` | `conv2_bias_rom` | 32 INT32 biases |
| `conv3_weights_8.mif` | `conv3_weights_rom` | 18 432 INT8 weights, 1 per byte |
| `conv3_bias_32.mif` | `conv3_bias_rom` | 64 INT32 biases |

Weight layout: OHWI (output channel, kernel row, kernel column, input channel), extracted from
`model/bottle_cnn_int8.tflite`.
