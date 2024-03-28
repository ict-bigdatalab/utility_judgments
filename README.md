# utility_judgments

## GTI Benchmark 
### Download datsets with ground_truth evidence
Download NQ test data from [test data of NQ](https://ai.google.com/research/NaturalQuestions/download); Dev data of HotpotQA from [KILT](https://github.com/facebookresearch/KILT); dev data of MSMACRO from [msmarco](https://microsoft.github.io/msmarco/) and [msm-qa](https://github.com/microsoft/MSMARCO-Question-Answering)

### Dense retrieval
We directly use [RocketQAv2](https://github.com/PaddlePaddle/RocketQA) on wiki-based NQ and HotpotQA datsets and [ADORE](https://github.com/jingtaozhan/DRhard) on web-based MSMARCO dataset.

### Counterfactual passages (CP)
We use the [entity substition](https://github.com/apple/ml-knowledge-conflicts) and [generation method](https://github.com/OSU-NLP-Group/LLM-Knowledge-Conflict)

### Highly relevant noisy passages (HRNP) and Weakly relevant noisy passages (WRNP)
Filter out results from existing retrievers that do not contain answers, and the reference is [noisy passages](https://github.com/RUCAIBox/LLM-Knowledge-Boundary)

### Candidate passages construction

We have also provided the final GTI benchmark, which you can download from [link](https://drive.google.com/drive/folders/1zmj2QiAxqsNfDf7iihYYKsdhL-WbvAYb?usp=drive_link)

## GTU benchmark

We have also provided the final GTI benchmark, which you can download from [link](https://drive.google.com/drive/folders/1zmj2QiAxqsNfDf7iihYYKsdhL-WbvAYb?usp=drive_link)



## Utility judgments of LLMs
Taking the testing of LlaMa 2-13B as an example, we demonstrated the use of four methods: pointwise, pairwise, list wise set, and list wise rank. If you want to test other models, you can directly replace them.

```
python llama2-point.py

```











