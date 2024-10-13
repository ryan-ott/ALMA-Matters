SLICE=${1}
TEST_PAIRS=${2}


set +e

## Evaluation
for pair in ${TEST_PAIRS//,/ }; do
    src=$(echo ${pair} | cut -d "-" -f 1)
    tgt=$(echo ${pair} | cut -d "-" -f 2)
    TOK="13a"
    if [ ${tgt} == "zh" ]; then
        TOK="zh"
    elif [ ${tgt} == "ja" ]; then
        TOK="ja-mecab"
    fi
    echo "--------------------Results for ${pair}-------------------------------------"
    # if [${SLICE} == "Baseline"]; then
    #     src_path=results/Baseline/result_${src}-${tgt}.txt
    #     output_path=results/Baseline/result_${src}-${tgt}.txt
    # else
    #     src_path=results/${SLICE}-slice/result_${src}-${tgt}.txt
    #     output_path=results/${SLICE}-slice/result_${src}-${tgt}.txt
    # fi
    src_path=results/${SLICE}-slice/result_${src}-${tgt}.txt
    # tgt_path=./datasets/human-translations/ALMA_test_${src}${tgt}.json
    if [[ -f "../datasets/human-translations/ALMA_test_${src}-${tgt}.txt" ]]; then
        echo "Target text file already exists. Using existing file."
    else
        touch "../datasets/human-translations/ALMA_test_${src}-${tgt}.txt"
        # jq -r --arg key "$tgt" '.[].translation[$tgt]' "../datasets/human-translations/ALMA_test_${src}-${tgt}.json" > "../datasets/human-translations/ALMA_test_${src}-${tgt}.txt"
        python - <<END
import json
# Load the JSON file
with open("../datasets/human-translations/ALMA_test_${src}-${tgt}.json", 'r', encoding='utf-8') as f:
    data = json.load(f)


# Extract values for the specified key
with open("../datasets/human-translations/ALMA_test_${src}-${tgt}.txt", "w", encoding='utf-8') as out_file:
    for item in data:
        out_file.write(item["translation"]["$tgt"] + "\n")


END
        echo "Values have been written to ../datasets/human-translations/ALMA_test_${src}-${tgt}.txt"
    fi
    tgt_path=../datasets/human-translations/ALMA_test_${src}-${tgt}.txt
    # tgt_path=results/baseline/result_${src}-${tgt}.txt


    # cp ${src_path} ${OUTPUT_DIR}


    # output_path=${OUTPUT_DIR}/result_${src}-${tgt}.txt
    output_path=results/${SLICE}-slice/result_${src}-${tgt}.txt
    # cp ${src_path} ${output_path}

    # Baseline results
    src_path=results/Baseline/result_${src}-${tgt}.txt
    output_path=results/Baseline/result_${src}-${tgt}.txt

    # # qlora base results
    # src_path=results/qlora_base/result_${src}-${tgt}.txt
    # output_path=results/qlora_base/result_${src}-${tgt}.txt
 
    # qlora base16 results
    # src_path=results/qlora_base_16/result_${src}-${tgt}.txt
    # output_path=results/qlora_base_16/result_${src}-${tgt}.txt

    echo "Source Path: $src_path"
    echo "Output Path: $output_path"
    echo "Target Path: $tgt_path"


    SACREBLEU_FORMAT=text sacrebleu -tok ${TOK} -w 2 ${tgt_path} < ${src_path} > ${output_path}.bleu
    cat ${output_path}.bleu
    comet-score -s ${src_path} -t ${output_path} -r ${tgt_path} --batch_size 256 --model Unbabel/wmt22-comet-da --gpus 1 > ${output_path}.comet
    # comet-score -s ${src_path} -t ${output_path} --batch_size 256 --model Unbabel/wmt22-cometkiwi-da --gpus 1 > ${output_path}.cometkiwi
    # comet-score -s ${src_path} -t ${output_path} --batch_size 8 --model Unbabel/wmt23-cometkiwi-da-xxl --gpus 1 > ${output_path}.cometkiwi_10b
    # comet-score -s ${src_path} -t ${output_path} --batch_size 8 --model Unbabel/XCOMET-XXL --gpus 1 --to_json ${output_path}.xcomet.output.json > ${output_path}.xcomet_10b    
    tail -n 1 ${output_path}.comet
done


for pair in ${TEST_PAIRS//,/ }; do
    src=$(echo ${pair} | cut -d "-" -f 1)
    tgt=$(echo ${pair} | cut -d "-" -f 2)
    echo "---------------------------${src}-${tgt}-------------------------------"
    output_path=${OUTPUT_DIR}/test-${src}-${tgt}
    cat ${output_path}.bleu
    tail -n 1 ${output_path}.comet
    # tail -n 1 ${output_path}.cometkiwi
    # tail -n 1 ${output_path}.cometkiwi_10b
    # tail -n 2 ${output_path}.xcomet_10b
done
