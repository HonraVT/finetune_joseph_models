import json
import random
import re
from collections import defaultdict


def load_file(file: str):
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_file(file: str, state):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4, ensure_ascii=False)


def normalize(file_path_in, file_path_out):
    data = load_file(file_path_in)["threads"]
    dataset = []
    for entry in data:
        post = entry["post"].strip()
        if post:
            quote = entry["quote"].strip()
            dataset.append({"input": quote if quote else entry["title"].strip(), "output": post})
    save_file(file_path_out, dataset)


def replace_words(file_path_in, file_path_out):
    words_to_replace = [["SerDeLuz", "mulher"], ["SeresDeLuz", "mulheres"], ["lâmpada", "mulher"],
        ["lâmpadas", "mulheres"],

    ]
    data = load_file(file_path_in)
    for entry in data:
        for old_word, new_word in words_to_replace:
            entry["input"] = re.sub(rf"\b{old_word}\b", new_word, entry["input"], flags=re.IGNORECASE)
            entry["output"] = re.sub(rf"\b{old_word}\b", new_word, entry["output"], flags=re.IGNORECASE)
    save_file(file_path_out, data)


def contains_keyword(text, keyword):
    return keyword.lower() in text.lower()


def filter_data(input_file, output_file, keywords_list, limit=100, save=True):
    data = load_file(input_file)
    print("dataset length:", len(data))

    keyword_counts = defaultdict(list)

    for entry in data:
        for keyword in keywords_list:
            if contains_keyword(entry["input"], keyword) or contains_keyword(entry["output"], keyword):
                if len(keyword_counts[keyword]) < limit:
                    keyword_counts[keyword].append(entry)
                break

    filtered_data = []
    for entries in keyword_counts.values():
        filtered_data.extend(entries)

    print("entries:")
    for i in filtered_data:
        print(i)
    print("length:", len(filtered_data))

    if save:
        save_file(output_file, filtered_data)


def randomize(input_file: str, output_file: str):
    data = load_file(input_file)
    random.shuffle(data)
    save_file(output_file, data)
    print(f"Arquivo randomizado: {output_file}")


def trim_list(input_file, output_file, size):
    data = load_file(input_file)
    data = data[:size]
    save_file(output_file, data)


jose_keywords = ["ônibus", "onibus", "sims", "Zelão", "José", "Jose", "Cristione", "Milena", "nicole", "Beto",
                 "Busologo", "Busólogo", "jogo", "carro", "mobi", "dirigir", "CNH", "trabalho""mulher", "trabalho",
                 "doente", "trata", "respeit"]

normalize("fiat_mobi_posts_final.json", "fiat_mobi_data.json")
# replace_words("fiat_mobi_data.json", "fiat_mobi_data.json")
# filter_data("jose_data_merged_randomized.json", "jose_dataset_2405.json", jose_keywords, limit=170, save=True)
# randomize("jose_data_merged.json", "jose_data_merged_randomized.json")
# trim_list("jose_data_merged_randomized.json", "jose_data_merged_randomized_trimmed.json", 2500)
