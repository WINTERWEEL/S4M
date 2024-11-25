# S4M

## Usage

#### Setup

Suppose you have Anaconda installed to setup environment. First, please install packages using provided `environment.yaml`.

```shell
cd S4M
conda env create -f environment.yaml
conda activate S4M
```

#### Dataset

We show the example of running S4M on our benchmark dataset weather with missing observations.

#### Training

The best checkpoints will be saved in `checkpoint` directory. 

```shell
python run.py --is_training 1 --root_path ./data/weather/chunk_missing/ --data_path weather_missing.csv --model_id weather_192_96 --model S4M --per_mem_size 5 --en_conv_hidden_size 256 --data custom4 --topK 20 --thres1 0.95 --thres2 0.6 --M 50 --memory_size 200 --momentum 0.99 --short_len 32 --features M --seq_len 192 --pred_len 96 --e_layers 8 --enc_in 21 --dec_in 21 --c_out 21 --des Exp --d_model 256 --learning_rate 0.005 --train_epochs 15 --num_pattn 30 --num_dG 1 --encoder_type 3 --memnet 13 --mem_type 3 --d_ff 256 --itr 1 --mean_type global_mean --fillna mean
```
