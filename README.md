# Deep Patent Landscaping
## Patent Landscaping
This is a implementation of deep patent landscaping introduced in the paper ["Deep Patent landscaping Model Using the Transformer and Graph Embedding"](https://arxiv.org/abs/1903.05823)

Among the vast patent data, selecting patents related to specific technology topics is a challenge. This task mainly is done by experts in the patent field and is very inefficient and expensive. The patent landscaping model using deep neural nets can be a clue to solving this problem.

## Dataset
We used search formula from [KISTA](http://biz.kista.re.kr/patentmap/front/common.do?method=main) and converted it to queries for BigQuery. Obtained data is too large, so we sampled data using heuristic method for training. You can download the full dataset and sampled data [here](https://docs.google.com/document/d/1yMS7yXQbTdQ9V_2ZfoYfr3iASI7ppkPJJIB0G_pH3Ho/edit).

| Acronyms | Full Name |
|:--:|:--:|
| MPUART | Marine Plant Using Augmented Reality Technology |
| 1MWDFS | Technology for 1MW Dual Frequency System |
| MRRG | Technology for Micro Radar Rain Gauge |
| GOCS | Technology for Geostationary Orbit Complex Satellite |

## Pretrained Embedding
We used Word2Vec for abstract embedding and Diff2Vec for classification code embedding. You can download the pretrained embedding [here](https://docs.google.com/document/d/1yMS7yXQbTdQ9V_2ZfoYfr3iASI7ppkPJJIB0G_pH3Ho/edit).
- pretrained word2vec location
  - /embeddings/w2v
- pretrained diff2vec location
  - /embeddings/graph

## Requirements
Tensorflow==1.x  
Keras==2.3.1  
pandas==0.25.3  
scikit-learn==0.21.3  

## Run
```bash
python train.py --data mrrg
```
