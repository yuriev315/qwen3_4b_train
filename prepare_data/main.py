from download_jsonl_gz import download_jsonl_gz
from zip2json import to_json
from gen_sft_dataset import generate_sft_data
from gen_dpo_dataset import generate_dpo_data
from download_zip import download_all_eval_zips
from sft_train import sft_train
from dpo_train import dpo_train
from huggingface_hub import login
login(token="hf_************************")

if __name__ == '__main__':
    data_download_flag = False
    if data_download_flag:
        download_jsonl_gz()
        download_all_eval_zips()

    data_process_flag = False
    if data_process_flag:
        to_json()
        generate_sft_data()
        generate_dpo_data()
    sft_train_flag = True
    if sft_train_flag:
        sft_train()
    dpo_train_flag = False
    if dpo_train_flag:
        dpo_train()

