def get_sentencepiece_model_for_beit3():
    from transformers import XLMRobertaTokenizer
    return XLMRobertaTokenizer("../beit3.spm")

tokenizer = get_sentencepiece_model_for_beit3()

def _write_data_into_jsonl(items, jsonl_file):
    with open(jsonl_file, mode="w", encoding="utf-8") as writer:
        for data in items:
            writer.write(json.dumps(data, indent=None))
            writer.write('\n')
    print("Write %s with %d items !" % (jsonl_file, len(items)))

import json
from tqdm import tqdm
from datasets import load_dataset
dataset_path = "../datasets/ATOMIC/"
splits = ["train", "validation", "test"] #["train", "dev", "test"]
for split in splits:
    items = []

    # with open(f"../../datasets/ATOMIC/Atomic_text_{split}.json", "r") as reader:
    #     for line in reader:
    #         data = json.loads(line)
    #         ann.append(data)
    ann = load_dataset("TREC-AToMiC/AToMiC-Texts-v0.2.1", split=split,
                                      num_proc=4)
    qrels = load_dataset("TREC-AToMiC/AToMiC-Qrels-v0.2", split=split)

    id2pos = {}
    for pos, dict in tqdm(enumerate(ann), total=len(ann)):
        id2pos[dict['text_id']] = pos


    for idx in tqdm(range(len(qrels))):

        new_dict = {}
        text_id = qrels[idx]['text_id']
        data = ann[id2pos[text_id]]

        context_page_des_tokens = tokenizer.tokenize(data[idx]["context_page_description"])
        context_sec_des_tokens = tokenizer.tokenize(data[idx]["context_section_description"])
        page_title_tokens = tokenizer.tokenize(data[idx]["page_title"])
        section_title_tokens = tokenizer.tokenize(data[idx]["section_title"])
        category_tokens = tokenizer.tokenize(''.join(data[idx]["category"]))


        context_page_des_tokens_ids = tokenizer.convert_tokens_to_ids(context_page_des_tokens)
        context_sec_des_tokens_ids = tokenizer.convert_tokens_to_ids(context_sec_des_tokens)
        page_title_tokens_ids = tokenizer.convert_tokens_to_ids(page_title_tokens)
        section_title_tokens_ids = tokenizer.convert_tokens_to_ids(section_title_tokens)
        category_tokens_ids = tokenizer.convert_tokens_to_ids(category_tokens)

        # new_dict["context_page_des_tokens"] = context_page_des_tokens
        # new_dict["context_sec_des_tokens"] = context_sec_des_tokens
        # new_dict["page_title_tokens"] = page_title_tokens
        # new_dict["section_title_tokens"] = section_title_tokens
        # new_dict["category_tokens"] = category_tokens
        new_dict["context_page_description_tokens_ids"] = context_page_des_tokens_ids
        new_dict["context_section_description_tokens_ids"] = context_sec_des_tokens_ids
        new_dict["page_title_tokens_ids"] = page_title_tokens_ids
        new_dict["section_title_tokens_ids"] = section_title_tokens_ids
        new_dict["category_tokens_ids"] = category_tokens_ids

        items.append(new_dict)

    _write_data_into_jsonl(items, f"../../datasets/ATOMIC/Atomic_text_tokenized_{split}_all_fields.json")