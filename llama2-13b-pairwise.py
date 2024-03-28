import os
import sys
sys.path.append(".")
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
os.environ["WORLD_SIZE"] = "1"
from tqdm import tqdm
import json
import torch
import logging
import argparse
from work.utils.utils import load_source
from work.utils.prompt import get_prompt_multi_docs_all_pair
from vllm import LLM, SamplingParams
import transformers
def get_args(ra_dict, order, file_type):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='work/dataset/GTI/'+file_type+order+'.json')
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--model_dir', default="llama2-13b-chat")
    parser.add_argument('--type', type=str, default='llama2') ## prompt type
    parser.add_argument('--ra', type=str, default="none", choices=ra_dict.keys())
    parser.add_argument('--outfile', type=str, default='data/method/'+file_type+'-' + order + '.json')
    args = parser.parse_args()
    args.ra = ra_dict
    return args


def main(ra_dict, order, file_type, llm, tokenizer, sampling_params):
    args = get_args(ra_dict, order, file_type)
    path = 'work/dataset/pairwise/llama2-13b_none/'
    args.outfile = path+file_type+ "_" +order + '.json'
    if not os.path.exists(path):
        os.makedirs(path)
    begin = 0
    if os.path.exists(args.outfile):
        outfile = open(args.outfile, 'r', encoding='utf-8')
        for line in outfile.readlines():
            if line != "":
                begin += 1
        outfile.close()
        outfile = open(args.outfile, 'a', encoding='utf-8')
    else:
        outfile = open(args.outfile, 'w', encoding='utf-8')

    all_data = load_source(args.source)  ## read the file and load it
    num_output = 0
    num = 0
    all_num = 0
    print("begin:", begin)
    try:
        instruct = "Please tell me which passage would contribute more utility to answering the above question, like 'My choice: Passage-0 or Passage-1'."
        with torch.no_grad():
            for sample in tqdm(all_data[begin:], desc="Filename: %s" % args.outfile):
                passages = sample["passage"]
                question = sample["question"]
                number = len(sample["passage"])
                passages_types = sample["passages_types"]
                labels = sample["ground_truth_label"]
                big = set()
                prompts_list = []
                for i in range(len(passages) - 1):
                    for j in range(i + 1, len(passages)):
                        pair = "\nPassage-0: " + passages[i] + "\n" + "Passage-1: " + passages[j]
                        prompts = "Given a question: {question}, two passages: {pair} \n" + "\n" + instruct
                        prompts = prompts.format(pair=pair, question=question)
                        message_prompt = "Given a question and two passages, please help me determine which passage would contribute more utility to answering the above question." + "\n" + instruct
                        prompts = "<s>[INST] <<SYS>>"+message_prompt+"<</SYS>>" + prompts+"[/INST]"
                        prompts_list.append(prompts)
                outputs = llm.generate(prompts_list, sampling_params)
                if len(outputs) != len(prompts_list):
                    outputs = []
                    for prompts in prompts_list:
                        output = llm.generate([prompts], sampling_params)
                        outputs += [output[0]]
                assert len(outputs) == len(prompts_list)
                ress = []  
                for output in outputs:
                    generated_text = output.outputs[0].text
                    res = generated_text
                    print(res)
                    ress.append(res)
                    if res is None:
                        print(0)
                        big.add((i, j))
                        continue
                    if "choice" in res:
                        res = res.split("choice")[1].split(".")[0]
                    else:
                        res = res
                    if "passage-0" in res.lower():
                        print("passage-0")
                        big.add((i, j))
                    else:
                        print("passage-1")
                        big.add((j, i))
                print("model_out:", res)
                print('ground_truth:', labels)
                outfile.write(json.dumps({
                    "passages_types": passages_types,
                    "question": question,
                    "prompt": prompts,
                    "passage": passages,
                    "output_all": ress,
                    "LLM_output_all": list(big),
                    "ground_truth_label": labels,
                    "message_prompt": message_prompt
                }) + "\n")
    except Exception as e:
        logging.exception(e)

    finally:
        print(args.outfile, " has output %d line(s)." % num_output)
        outfile.close()



if __name__ == '__main__':
    ra_dict = {
        'gold_ctxs': 10,
        'strong_ctxs': 10,
        'weak_ctxs': 10,
        'type_swap_answer': 10,
        'popularity_answer': 10,
        'corpus_swap_answer': 10,
        'alias_answer': 10
    }
    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)
    tokenizer = transformers.AutoTokenizer.from_pretrained("models/llama2-13b-chat/")
    llm = LLM(model="models/llama2-13b-chat/")
    main(ra_dict, "random", "MSM", llm, tokenizer, sampling_params)
    main(ra_dict, "random", "NQ", llm, tokenizer, sampling_params)
    main(ra_dict, "random", "HQ", llm, tokenizer, sampling_params)


    






    



