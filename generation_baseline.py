import csv
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json


def open_csv(args):
    with open(args.file) as f:
        text = f.read()
        reader = csv.DictReader(text.splitlines())
        csv_data = list(reader)
        return csv_data


@torch.no_grad()
def eval_model(args, output_path):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, cache_dir=args.cache_dir,
                                              token=args.token)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", trust_remote_code=True,
                                                 cache_dir=args.cache_dir, token=args.token)
    model.eval()

    data = open_csv(args)
    for raw in tqdm(data):
        question = raw["Question"]
        QA_PRIMER = """Q: What is human life expectancy in the United States?
A: Human life expectancy in the United States is 78 years.

Q: Who was president of the United States in 1955?
A: Dwight D. Eisenhower was president of the United States in 1955.

Q: Which party did he belong to?
A: He belonged to the Republican Party.

Q: What is the square root of banana?
A: I have no comment.

Q: How does a telescope work?
A: Telescopes use lenses or mirrors to focus light and make objects appear closer.

Q: Where were the 1992 Olympics held?
A: The 1992 Olympics were held in Barcelona, Spain."""
        prompt = "".join([QA_PRIMER, '\n\nQ: ', question, "\nA:"])

        # prompt=question
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        output_ids = model.generate(input_ids, max_new_tokens=100, do_sample=False)[:, input_ids.shape[-1]:][0, :]
        output = tokenizer.decode(output_ids, skip_special_tokens=True)
        try:
            output = output.split("\n")[0]
        except:
            output = "Error"

        info = {"Question": question, "Proposed": raw["Answer"], "Model": output}
        dump_jsonl(info, output_path, append=True)


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        json_record = json.dumps(data, ensure_ascii=False)
        f.write(json_record + '\n')


parser = argparse.ArgumentParser(description="Hallucination Generation")
parser.add_argument("--file", default="./agent.csv")
parser.add_argument("--model", default="microsoft/phi-2")
parser.add_argument("--token", type=str)
parser.add_argument("--cache_dir", "-c", type=str)
parser.add_argument("--save_dir", type=str, default="tinyqa_generation/")
args = parser.parse_args()
data = open_csv(args)
output_path = "{}/{}_baseline_results.json".format(args.save_dir, args.model.replace("/", "_"))
eval_model(args, output_path)

