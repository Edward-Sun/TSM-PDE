# Reproduction

## Reproducing Direct Numerical Simulation (DNS) Results

For DNS-2048x2048, run the following command:

```bash
export PSCRATCH=/path/to/your/scratch/directory
export TRAINDATA="models/dns_2048x2048_train*/train.nc.hdf5-*"
export PREDICTDATA="models/dns_2048x2048_test/test.nc.hdf5-*"
export STORAGE_PATH=$PSCRATCH/cfd
export MODEL_NAME=dns_64

python -u tsm/train.py \
  --model_input_size=2048 \
  --model_encode_steps=1 \
  --model_decode_steps=32 \
  --model_predict_steps=32 \
  --delta_time=0.007012483601762931 \
  --train_split="$STORAGE_PATH/$TRAINDATA" \
  --predict_split="$STORAGE_PATH/$PREDICTDATA" \
  --train_device_batch_size=4 \
  --train_lr_init=0.001 \
  --train_lr_warmup_epochs=0 \
  --train_log_every=1 \
  --no_train \
  --do_predict \
  --no_dropout \
  --predict_simulation_time=25.0 \
  --explicit_inner_steps=8 \
  --output_dir="$STORAGE_PATH/models/$MODEL_NAME" \
  --gin_file="tsm/configs/implicit_diffusion_dns_config.gin" \
  --gin_file="tsm/configs/kolmogorov_forcing.gin"
```


For DNS-64x64, run the following command:

```bash
export PSCRATCH=/path/to/your/scratch/directory
export TRAINDATA="models/dns_2048x2048_train*/train.nc.hdf5-*"
export PREDICTDATA="models/dns_2048x2048_test/test.nc.hdf5-*"
export PREDICTTARGET="models/dns_2048/predict.nc"
export STORAGE_PATH=$PSCRATCH/cfd
export MODEL_NAME=dns_64

python -u tsm/train.py \
  --model_input_size=64 \
  --model_encode_steps=1 \
  --model_decode_steps=32 \
  --model_predict_steps=32 \
  --delta_time=0.007012483601762931 \
  --train_split="$STORAGE_PATH/$TRAINDATA" \
  --predict_split="$STORAGE_PATH/$PREDICTDATA" \
  --predict_target="$STORAGE_PATH/$PREDICTTARGET" \
  --train_device_batch_size=4 \
  --train_lr_init=0.001 \
  --train_lr_warmup_epochs=0 \
  --train_log_every=1 \
  --no_train \
  --do_predict \
  --no_dropout \
  --predict_simulation_time=25.0 \
  --explicit_inner_steps=8 \
  --output_dir="$STORAGE_PATH/models/$MODEL_NAME" \
  --gin_file="tsm/configs/implicit_diffusion_dns_config.gin" \
  --gin_file="tsm/configs/kolmogorov_forcing.gin"
```

## Reproducing LI Results

For [Learned Interpolator (LI)](https://www.pnas.org/doi/10.1073/pnas.2101784118), run the following command:

```bash
export PSCRATCH=/path/to/your/scratch/directory
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HAIKU_FLATMAPPING=0

export TRAINDATA="models/dns_2048x2048_train*/train.nc.hdf5-*"
export PREDICTDATA="models/dns_2048x2048_test/train.nc.hdf5-*"
export PREDICTTARGET="models/my_dns_2048/predict.nc"
export STORAGE_PATH=$PSCRATCH/cfd
export MODEL_NAME=li_64

mkdir -p $STORAGE_PATH/models/$MODEL_NAME

python -u tsm/train.py \
  --model_encode_steps=32 \
  --model_decode_steps=32 \
  --model_predict_steps=32 \
  --train_device_batch_size=8 \
  --delta_time=0.007012483601762931 \
  --train_split="$STORAGE_PATH/$TRAINDATA" \
  --predict_split="$STORAGE_PATH/$PREDICTDATA" \
  --predict_target="$STORAGE_PATH/$PREDICTTARGET" \
  --train_lr_init=0.001 \
  --train_lr_warmup_epochs=0.05 \
  --train_epochs=4.0 \
  --adam_beta2=0.98 \
  --train_log_every=100 \
  --resume_checkpoint \
  --train_predict_every=1000 \
  --inner_steps=1 \
  --explicit_inner_steps 8 \
  --do_predict \
  --mp_skip_nonfinite \
  --output_dir="$STORAGE_PATH/models/$MODEL_NAME" \
  --gin_file="tsm/configs/generalized_li_config.gin" \
  --gin_file="tsm/configs/tsm_kolmogorov_forcing.gin" \
  --gin_param="fixed_scale.rescaled_one = 0.1" \
  --gin_param="tsm_forward_tower_factory.num_hidden_channels = 256" \
  --gin_param="tsm_forward_tower_factory.num_hidden_layers = 6" \
  --gin_param="tsm_aligned_array_encoder.n_frames = 1" \
  --gin_param="tsm_forward_tower_factory.dropout_rate = 0.0"
```

To evaluate the pre-trained model checkpoints on the test set, run the following command:

```bash
export CKPT_PATH=/path/to/your/pretrained_checkpoint
export PSCRATCH=/path/to/your/scratch/directory
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HAIKU_FLATMAPPING=0

export TRAINDATA="models/dns_2048x2048_train*/train.nc.hdf5-*"
export PREDICTDATA="models/dns_2048x2048_test/train.nc.hdf5-*"
export PREDICTTARGET="models/my_dns_2048/predict.nc"
export STORAGE_PATH=$PSCRATCH/cfd
export MODEL_NAME=li_64

mkdir -p $STORAGE_PATH/models/$MODEL_NAME

python -u tsm/train.py \
  --model_encode_steps=32 \
  --model_decode_steps=32 \
  --model_predict_steps=32 \
  --train_device_batch_size=8 \
  --delta_time=0.007012483601762931 \
  --train_split="$STORAGE_PATH/$TRAINDATA" \
  --predict_split="$STORAGE_PATH/$PREDICTDATA" \
  --predict_target="$STORAGE_PATH/$PREDICTTARGET" \
  --train_lr_init=0.001 \
  --train_lr_warmup_epochs=0.05 \
  --train_epochs=4.0 \
  --adam_beta2=0.98 \
  --train_log_every=100 \
  --resume_checkpoint \
  --train_predict_every=1000 \
  --inner_steps=1 \
  --explicit_inner_steps 8 \
  --do_predict \
  --mp_skip_nonfinite \
  --output_dir="$STORAGE_PATH/models/$MODEL_NAME" \
  --gin_file="tsm/configs/generalized_li_config.gin" \
  --gin_file="tsm/configs/tsm_kolmogorov_forcing.gin" \
  --gin_param="fixed_scale.rescaled_one = 0.1" \
  --gin_param="tsm_forward_tower_factory.num_hidden_channels = 256" \
  --gin_param="tsm_forward_tower_factory.num_hidden_layers = 6" \
  --gin_param="tsm_aligned_array_encoder.n_frames = 1" \
  --gin_param="tsm_forward_tower_factory.dropout_rate = 0.0" \
  --resume_checkpoint_dir="$CKPT_PATH" \
  --explicit_resume \
  --no_train
```

## Reproducing TSM Results

For [Temporal Stencil Modeling (TSM)](https://arxiv.org/pdf/2302.08105.pdf), run the following command:

```bash
export CKPT_PATH=/path/to/your/pretrained_checkpoint
export PSCRATCH=/path/to/your/scratch/directory
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HAIKU_FLATMAPPING=0

export TRAINDATA="models/dns_2048x2048_train*/train.nc.hdf5-*"
export PREDICTDATA="models/dns_2048x2048_test/train.nc.hdf5-*"
export PREDICTTARGET="models/my_dns_2048/predict.nc"
export STORAGE_PATH=$PSCRATCH/cfd
export MODEL_NAME=tsm_64_prefix32_hippo_dt1.0_tb4

mkdir -p "$STORAGE_PATH"/models/$MODEL_NAME

python -u tsm/train.py \
  --model_encode_steps=32 \
  --model_decode_steps=32 \
  --model_predict_steps=32 \
  --train_device_batch_size=8 \
  --delta_time=0.007012483601762931 \
  --train_split="$STORAGE_PATH/$TRAINDATA" \
  --predict_split="$STORAGE_PATH/$PREDICTDATA" \
  --predict_target="$STORAGE_PATH/$PREDICTTARGET" \
  --train_lr_init=0.001 \
  --train_lr_warmup_epochs=0.05 \
  --train_epochs=8.0 \
  --adam_beta2=0.98 \
  --train_log_every=100 \
  --train_predict_every=1000 \
  --inner_steps=1 \
  --explicit_inner_steps 8 \
  --do_predict \
  --mp_skip_nonfinite \
  --output_dir="$STORAGE_PATH/models/$MODEL_NAME" \
  --gin_file="tsm/configs/generalized_li_config.gin" \
  --gin_file="tsm/configs/tsm_kolmogorov_forcing.gin" \
  --gin_param="fixed_scale.rescaled_one = 0.1" \
  --gin_param="tsm_forward_tower_factory.num_hidden_channels = 256" \
  --gin_param="tsm_forward_tower_factory.num_hidden_layers = 6" \
  --gin_param="tsm_aligned_array_encoder.n_frames = 32" \
  --gin_param="tsm_forward_tower_factory.dropout_rate = 0.0" \
  --gin_param="TSMModularStepModel.temporal_bundle_steps = 4" \
  --gin_param="TSMModularStepModel.hippo_hidden_size = 256" \
  --gin_param="JaxAdaptiveTransition.dt = 1.0"
```

To evaluate the pre-trained model checkpoints on the test set, run the following command:

```bash
export PSCRATCH=/path/to/your/scratch/directory
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HAIKU_FLATMAPPING=0

export TRAINDATA="models/dns_2048x2048_train*/train.nc.hdf5-*"
export PREDICTDATA="models/dns_2048x2048_test/train.nc.hdf5-*"
export PREDICTTARGET="models/my_dns_2048/predict.nc"
export STORAGE_PATH=$PSCRATCH/cfd
export MODEL_NAME=tsm_64_prefix32_hippo_dt1.0_tb4

mkdir -p "$STORAGE_PATH"/models/$MODEL_NAME

python -u tsm/train.py \
  --model_encode_steps=32 \
  --model_decode_steps=32 \
  --model_predict_steps=32 \
  --train_device_batch_size=8 \
  --delta_time=0.007012483601762931 \
  --train_split="$STORAGE_PATH/$TRAINDATA" \
  --predict_split="$STORAGE_PATH/$PREDICTDATA" \
  --predict_target="$STORAGE_PATH/$PREDICTTARGET" \
  --train_lr_init=0.001 \
  --train_lr_warmup_epochs=0.05 \
  --train_epochs=8.0 \
  --adam_beta2=0.98 \
  --train_log_every=100 \
  --train_predict_every=1000 \
  --inner_steps=1 \
  --explicit_inner_steps 8 \
  --do_predict \
  --mp_skip_nonfinite \
  --output_dir="$STORAGE_PATH/models/$MODEL_NAME" \
  --gin_file="tsm/configs/generalized_li_config.gin" \
  --gin_file="tsm/configs/tsm_kolmogorov_forcing.gin" \
  --gin_param="fixed_scale.rescaled_one = 0.1" \
  --gin_param="tsm_forward_tower_factory.num_hidden_channels = 256" \
  --gin_param="tsm_forward_tower_factory.num_hidden_layers = 6" \
  --gin_param="tsm_aligned_array_encoder.n_frames = 32" \
  --gin_param="tsm_forward_tower_factory.dropout_rate = 0.0" \
  --gin_param="TSMModularStepModel.temporal_bundle_steps = 4" \
  --gin_param="TSMModularStepModel.hippo_hidden_size = 256" \
  --gin_param="JaxAdaptiveTransition.dt = 1.0" \
  --resume_checkpoint \
  --resume_checkpoint_dir="$CKPT_PATH" \
  --explicit_resume \
  --no_train
```
