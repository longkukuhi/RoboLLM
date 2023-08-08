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
dataset_path = "../../datasets/ATOMIC/"
splits = ["train","validation"] #["train", "dev", "test"]

for split in splits:
    items = []
    ann = []
    with open(f"../datasets/ATOMIC/Atomic_text_{split}.json", "r") as reader:
        for line in reader:
            data = json.loads(line)
            ann.append(data)

    for idx in tqdm(range(len(ann))):
        new_dict = ann[idx]

        context_page_des_tokens = tokenizer.tokenize(new_dict["context_page_description"])
        context_sec_des_tokens = tokenizer.tokenize(new_dict["context_section_description"])
        page_title_tokens = tokenizer.tokenize(new_dict["page_title"])
        section_title_tokens = tokenizer.tokenize(new_dict["section_title"])
        category_tokens = tokenizer.tokenize(''.join(new_dict["category"]))


        context_page_des_tokens_ids = tokenizer.convert_tokens_to_ids(context_page_des_tokens)
        context_sec_des_tokens_ids = tokenizer.convert_tokens_to_ids(context_sec_des_tokens)
        page_title_tokens_ids = tokenizer.convert_tokens_to_ids(page_title_tokens)
        section_title_tokens_ids = tokenizer.convert_tokens_to_ids(section_title_tokens)

        new_dict["context_page_des_tokens"] = context_page_des_tokens
        new_dict["context_sec_des_tokens"] = context_sec_des_tokens
        new_dict["page_title_tokens"] = page_title_tokens
        new_dict["section_title_tokens"] = section_title_tokens
        new_dict["category_tokens"] = category_tokens
        new_dict["context_page_des_tokens_ids"] = context_page_des_tokens_ids
        new_dict["context_sec_des_tokens_ids"] = context_sec_des_tokens_ids
        new_dict["page_title_tokens_ids"] = page_title_tokens_ids
        new_dict["section_title_tokens_ids"] = section_title_tokens_ids

        items.append(new_dict)
    _write_data_into_jsonl(items, f"../datasets/ATOMIC/Atomic_text_tokenized_{split}.json")