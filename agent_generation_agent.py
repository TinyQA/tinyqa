import csv
from typing import Any
import argparse
from tqdm import tqdm
import json
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.agents import load_tools

def open_csv(args):
    with open(args.file,encoding="utf-8") as f:
        text=f.read()
        reader=csv.DictReader(text.splitlines())
        csv_data=list(reader)
        return csv_data
    

def pack_message(role: str, content: Any):
    return {"role": str(role), "content": content}

    
def eval_model(args,output_path):
    llm = ChatOpenAI(
        base_url=args.base_url,
        model=args.model,
        temperature=0
    )
    
    tools = load_tools(["ddg-search","llm-math"], llm=llm)
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    data = open_csv(args)
    for raw in tqdm(data):
        question = raw["Question"]
        nojump=True
        response=agent.invoke({"input":question})

        if nojump:
            info = {"Question":question,"Proposed":raw["Answer"],"Model":response['output']}
            dump_jsonl(info,output_path,append=True)
    

def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
            json_record = json.dumps(data, ensure_ascii=False)
            f.write(json_record+'\n')


parser = argparse.ArgumentParser(description="Generate")
parser.add_argument("--file",default="agent.csv")
parser.add_argument("--model",default="qwen2.5-3b-instruct")
parser.add_argument("--base_url",type=str)
parser.add_argument("--save_dir",type=str,default="tinyqa_agent/")
args = parser.parse_args()
data = open_csv(args)
output_path = "{}/{}_agent_results.json".format(args.save_dir, args.model.replace("/","_"))
eval_model(args,output_path)