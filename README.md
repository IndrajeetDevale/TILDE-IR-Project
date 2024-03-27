
This is the README for the project proposal based on the SIGIR2021 paper [TILDE: Term Independent Likelihood moDEl for Passage Re-ranking](http://ielab.io/publications/arvin-2021-TILDE).
For the original README provided by the authors of the paper, please refer to - https://github.com/ielab/TILDE/blob/main/README.md

# Clone the git repository of the TILDE code.
using the command `git clone https://github.com/ielab/TILDE.git`

# Install the dependencies
- h5py
- pytorch-lightning==1.9.0
- torch
- tqdm
- transformers
- datasets

# Install tensorboard and tensorboardXs
pip3 install -U tensorboard
pip3 install -U tensorboardX

# Download the MS MARCO dataset
Download `collection.tar.gz` from the MS MARCO passage ranking repository. Using the Command 
```
`wget -c --retry-connrefused --tries=0 --timeout=50 https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz⁠`
```
Unzip `collection.tar.gz` using the command `tar -xvf collection.tar.gz`

Now move the collection.tsv file into the folder `./data/collection`


## Passage re-ranking with TILDE
TILDE uses BERT to pre-compute passage representations. Since the MS MARCO passage collection has around 8.8m passages, it will require more than 500G to store the representations of the whole collection. To quickly try out TILDE, in this example, we only pre-compute passages that we need to re-rank.

### Indexing the collection

First, run the following command from the root:

```
python3 indexing.py \
--ckpt_path_or_name ielab/TILDE \
--run_path ./data/runs/run.trec2019-bm25.res \
--collection_path ./data/collection/collection.tsv \
--output_path ./data/index/TILDE
```

If you have a gpu with big memory, you can set `--batch_size` that suits your gpu the best.

This command will create a mini index in the folder `./data/index/TILDE` that stores representations of passages in the BM25 run file.

If you want to index the whole collection, simply run:

```
python3 indexing.py \
--ckpt_path_or_name ielab/TILDE \
--collection_path ./data/collection/collection.tsv \
--output_path ./data/index/TILDE
```

### Re-rank BM25 results.
After you got the index, now you can use TILDE to re-rank BM25 results.

Let‘s first check out what is the BM25 performance on TREC DL2019 with [trec_eval](https://github.com/usnistgov/trec_eval). 
Please follow the steps below for installation -
```
To install trec_eval, you can follow these steps:
1. Clone the repository at https://github.com/usnistgov/trec_eval to the same directory (TILDE).
2. Open a terminal or command prompt and navigate to the directory where you extracted the trec_eval source code.
3. Simply type 'make' in the terminal and press Enter. This command will compile the source code and create the trec_eval bina

```
Verification: After installation, you can verify that trec_eval is correctly installed by running trec_eval -h or trec_eval --help in the terminal. This should display the help information for trec_eval, indicating that it's installed and functioning properly.

Now, run:
```
/TILDE/trec_eval/trec_eval -m ndcg_cut.10 -m map ./data/qrels/2019qrels-pass.txt ./data/runs/run.trec2019-bm25.res
```
we get:

```
map                     all     0.3766
ndcg_cut_10             all     0.4973
```

Now run the command below to use TILDE to re-rank BM25 top1000 results:

```
python3 inference.py \
--run_path ./data/runs/run.trec2019-bm25.res \
--query_path ./data/queries/DL2019-queries.tsv \
--index_path ./data/index/TILDE/passage_embeddings.pkl \
--save_path ./data/runs/TILDE_alpha1.txt
```
It will generate another run file in `./data/runs/` and also will print the query latency of the average query processing time and re-ranking time:

```
Query processing time: 0.1 ms
passage re-ranking time: 4.8 ms
```

TILDE only uses 0.2ms to compute the query sparse representation and 6.7ms to re-rank 1000 passages retrieved by BM25. Note, by default, the code uses a pure query likelihood ranking setting (alpha=1).

Now let's evaluate the TILDE run:

```
/TILDE/trec_eval/trec_eval -m ndcg_cut.10 -m map ./data/qrels/2019qrels-pass.txt ./data/runs/TILDE_alpha1.txt
```
we get:

```
map                     all     0.4053
ndcg_cut_10             all     0.5737
```

If you want more improvement, you can interpolate query likelihood score with document likelihood by:

```
python3 inference.py \
--run_path ./data/runs/run.trec2019-bm25.res \
--query_path ./data/queries/DL2019-queries.tsv \
--index_path ./data/index/TILDE/passage_embeddings.pkl \
--alpha 0.5 \
--save_path ./data/runs/TILDE_alpha0.5.txt
```

you will get higher query latency:

```
Query processing time: 53.5 ms
passage re-ranking time: 12.0 ms
```
This is because now TILDE has an extra step of using deep language model to compute query dense representation. As a trade-off you will get higher effectiveness:

```
/TILDE/trec_eval/trec_eval -m ndcg_cut.10 -m map ./data/qrels/2019qrels-pass.txt ./data/runs/TILDE_alpha0.5.txt
```
we get:
```
map                     all     0.4199
ndcg_cut_10             all     0.6049
```

##
