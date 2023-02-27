# bash bash/run_post_ranking.sh -s 42 -r 1ep_0neg -o gpt-large_1_linear_1ep/output_dev_300 -c 2
while getopts ':c:o:r:' opt
do
    case $opt in
        c)
        CUDA_IDS="$OPTARG" ;;
        o)
        output_name="$OPTARG" ;;
        # m)
        # model_name="$OPTARG" ;;
        r)
        ranker_model="$OPTARG" ;;
        # s)
        # seed="$OPTARG" ;;
        ?)
        exit 1;;
    esac
done


precision=16
max_seq_length=128
batch_size=128
seed=42

data_dir=../data/simplified_dev_question.json
# data_dir=../data/simplified_test_question.json

# --model_name_or_path=${ranker_model}\
CUDA_VISIBLE_DEVICES=${CUDA_IDS} python run_post_ranking.py \
  --data_dir ${data_dir} \
  --model_name_or_path ../output/ranker_model_${ranker_model}/model/ \
  --answer_dir ../output/${output_name}/ \
  --batch_size 32 \
  --seed ${seed}