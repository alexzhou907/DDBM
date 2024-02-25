

DATASET_NAME=$1
PRED=$2
MODEL_PATH=$3
CHURN_STEP_RATIO=$4
GUIDANCE=$5
SPLIT=$6


source ./args.sh $DATASET_NAME $PRED 


N=40
GEN_SAMPLER=heun
BS=16
NGPU=4

mpiexec -n $NGPU python scripts/image_sample.py --exp=$EXP \
--batch_size $BS --churn_step_ratio $CHURN_STEP_RATIO --steps $N --sampler $GEN_SAMPLER \
--model_path $MODEL_PATH --attention_resolutions $ATTN  --class_cond False --pred_mode $PRED \
${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"}  \
${COND:+ --condition_mode="${COND}"} --sigma_data $SIGMA_DATA --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN --cov_xy $COV_XY \
--dropout 0.1 --image_size $IMG_SIZE --num_channels $NUM_CH --num_head_channels 64 --num_res_blocks $NUM_RES_BLOCKS \
--resblock_updown True --use_fp16 $USE_16FP --attention_type $ATTN_TYPE --use_scale_shift_norm True \
--weight_schedule bridge_karras --data_dir=$DATA_DIR \
 --dataset=$DATASET --rho 7 --upscale=False ${CH_MULT:+ --channel_mult="${CH_MULT}"} \
 ${UNET:+ --unet_type="${UNET}"} ${SPLIT:+ --split="${SPLIT}"} ${GUIDANCE:+ --guidance="${GUIDANCE}"}
 

