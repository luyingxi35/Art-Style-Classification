# Evaluation Results
## Structure implementation and validation
Baseline accuracy: 55%
> Cubism Accuracy: 55.00%  
Expressionism Accuracy: 15.00%  
Impressionism Accuracy: 30.00%  
Realism Accuracy: 90.00%  
Abstract Accuracy: 85.00%

Shallow NN accuracy: 60%
> Cubism Accuracy: 55.00%  
Expressionism Accuracy: 40.00%  
Impressionism Accuracy: 60.00%  
Realism Accuracy: 80.00%  
Abstract Accuracy: 65.00%

Hierarchy 1 (simply entropy): 59%
> Cubism Accuracy: 65.00%  
Expressionism Accuracy: 20.00%  
Impressionism Accuracy: 60.00%  
Realism Accuracy: 70.00%  
Abstract Accuracy: 80.00%

500 test data:  
Baseline: 51.8%
> Cubism Accuracy: 49.00%  
Expressionism Accuracy: 11.00%  
Impressionism Accuracy: 34.00%  
Realism Accuracy: 93.00%  
Abstract Accuracy: 72.00%

Shallow NN: 58.4%
> Total Accuracy: 58.40%  
Cubism Accuracy: 55.00% | Total: 100.0 | Correct: 55.0  
Expressionism Accuracy: 36.00% | Total: 100.0 | Correct: 36.0  
Impressionism Accuracy: 72.00% | Total: 100.0 | Correct: 72.0  
Realism Accuracy: 73.00% | Total: 100.0 | Correct: 73.0  
Abstract Accuracy: 56.00% | Total: 100.0 | Correct: 56.0  

| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Cubism        | 0.8871    | 0.5500 | 0.6790   | 100     |
| Expressionism | 0.3214    | 0.3600 | 0.3396   | 100     |
| Impressionism | 0.4800    | 0.7200 | 0.5760   | 100     |
| Realism       | 0.7087    | 0.7300 | 0.7192   | 100     |
| Abstract      | 0.7671    | 0.5600 | 0.6474   | 100     |
| **Accuracy**  |           |        | **0.5840**| 500     |
| **Macro Avg** | 0.6329    | 0.5840 | 0.5922   | 500     |
| **Weighted Avg** | 0.6329 | 0.5840 | 0.5922   | 500     |
 

Hierarchy 1 (simply entropy): 53.8%
> Cubism Accuracy: 50.00%  
Expressionism Accuracy: 23.00%  
Impressionism Accuracy: 63.00%  
Realism Accuracy: 74.00%  
Abstract Accuracy: 59.00%

Hierarchy 2 (select the max entropy): 58%
> Cubism Accuracy: 60.00%  
Expressionism Accuracy: 39.00%  
Impressionism Accuracy: 66.00%  
Realism Accuracy: 72.00%  
Abstract Accuracy: 53.00%

| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Cubism        | 0.7692    | 0.6000 | 0.6742   | 100     |
| Expressionism | 0.3824    | 0.3900 | 0.3861   | 100     |
| Impressionism | 0.5077    | 0.6600 | 0.5739   | 100     |
| Realism       | 0.6261    | 0.7200 | 0.6698   | 100     |
| Abstract      | 0.7067    | 0.5300 | 0.6057   | 100     |
| **Accuracy**  |           |        | **0.5800**| 500     |
| **Macro Avg** | 0.5984    | 0.5800 | 0.5819   | 500     |
| **Weighted Avg** | 0.5984 | 0.5800 | 0.5819   | 500     |

Reproduction guide:
- For baseline: change IS_BASELINE in evaluation.py into True
- For shallowNN: change IS_BASELINE in evaluation.py into False

## Patch Division
- No patch (baseline): 51.8%
> Cubism Accuracy: 49.00%  
Expressionism Accuracy: 11.00%  
Impressionism Accuracy: 34.00%  
Realism Accuracy: 93.00%  
Abstract Accuracy: 72.00%
- 4 patches: 58.8%
> Cubism Accuracy: 62.00%  
Expressionism Accuracy: 41.00%  
Impressionism Accuracy: 65.00%  
Realism Accuracy: 73.00%  
Abstract Accuracy: 53.00%

Classification Report (Precision / Recall / F1-Score):
| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
|  Cubism     |0.7949    |0.6200    |0.6966      | 100|
|Expressionism |    0.3981|    0.4100|    0.4039|       100|
|Impressionism |    0.5462|    0.6500|    0.5936|       100|
|      Realism |    0.5703|    0.7300|    0.6404|       100|
|     Abstract |    0.7361|    0.5300|    0.6163|       100|
| **accuracy** |           |         |     **0.5880**|       500|
|    macro avg  |   0.6091  |  0.5880   | 0.5902      | 500|
- 16 patches: 51%
> Cubism Accuracy: 50.00%  
Expressionism Accuracy: 15.00%  
Impressionism Accuracy: 70.00%  
Realism Accuracy: 60.00%  
Abstract Accuracy: 60.00%  

## Hierarchy structure
Add entropy-based fusion: 56%

Add MRF-optimization: No difference
- window 1 - 0.8: 56%
- window 2 - 0.5: 56%



