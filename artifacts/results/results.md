# Model Compression Results

Comparison of baseline, distilled, and quantized models (quality vs. inference cost).

| Model             | Device   |   Accuracy |   F1 (macro) |   Params (M) |   Size (MB) |   Latency bs=1 (ms) |   Throughput bs=32 (ex/s) |   Peak mem (MB) |
|-------------------|----------|------------|--------------|--------------|-------------|---------------------|---------------------------|-----------------|
| teacher           | cpu      |     0.9243 |       0.9243 |        109.5 |       417.7 |               28.29 |                      53.3 |          1388.8 |
| student-finetuned | cpu      |     0.8865 |       0.8864 |         67   |       256.3 |               14.43 |                     106.1 |          1676.9 |
| student-distilled | cpu      |     0.8819 |       0.8817 |         67   |       256.3 |               14.54 |                     103.6 |          1678   |
| student-quantized | cpu      |     0.8807 |       0.8807 |         67   |       133.2 |               32.41 |                      31.1 |          1460.4 |

_Latency/throughput measured with synthetic batches at seq_len from config; accuracy on the eval split. Device is shown because quantized PyTorch models run on CPU. Absolute numbers depend on your hardware._
