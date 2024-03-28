import os
import sys
sys.path.append(".")
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
os.environ["WORLD_SIZE"] = "1"
from tqdm import tqdm
import json
import logging
import argparse
from work.utils.utils import load_source
from vllm import LLM, SamplingParams
import transformers
def get_args(ra_dict, order, file_type):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='work/dataset/GTI/'+file_type+'random.json')
    parser.add_argument('--usechat', default=True)
    parser.add_argument('--type', type=str, default="") #list_rank_relevance, list_rank_useful
    parser.add_argument('--ra', type=str, default="none", choices=ra_dict.keys())
    parser.add_argument('--outfile', type=str, default='')
    args = parser.parse_args()
    args.ra = ra_dict
    return args

def main(ra_dict, order, file_type, llm, tokenizer, sampling_params):
    args = get_args(ra_dict, order, file_type)
    path = 'work/dataset/point_prompt2/'
    args.outfile = path + file_type + '-only-rea-' + order + '.json'
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
    exp = ""
    instruct = "Please first provide a brief reasoning you used to judge whether the passage has utility in answering the above question or not and tell me your overall judgment, like 'My judgment: yes/no'."
    try:
        for sample in tqdm(all_data[begin:len(all_data)], desc="Filename: %s" % args.outfile):
            passages = sample["passage"]
            question = sample["question"]
            passages_types = sample["passages_types"]
            labels = sample["ground_truth_label"]
            model_out_label = []
            prompts_list = []
            for i in range(len(passages)):
                pair = passages[i]
                prompts = "Given a question: {question}, a passage: {pair} \n" + exp + "\n" + instruct
                prompts = prompts.format(pair=pair, question=question)
                message_prompt = "Given a question and passage, please judge whether the passage has utility in answering the question." + exp + "\n" + instruct
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
                ress.append(res)
                if res is None:
                    print(0)
                    model_out_label.append(0)
                    continue
                print(res.lower())
                if ": yes" in res.lower():
                    print(1)
                    model_out_label.append(1)
                elif "is yes" in res.lower():
                    print(1)
                    model_out_label.append(1)
                elif " yes" in res.lower():
                    print(1)
                    model_out_label.append(1)
                else:
                    print(0)
                    model_out_label.append(0)
            outfile.write(json.dumps({
                "passages_types": passages_types,
                "question": sample["question"],
                "passage": passages,
                "prompts": prompts,
                "output_all": ress,
                "LLM_output_all": model_out_label,
                "ground_truth_label": labels
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
    main(ra_dict, "random", "NQ", llm, tokenizer, sampling_params)
    main(ra_dict, "random", "HQ", llm, tokenizer, sampling_params)
    main(ra_dict, "random", "MSM", llm, tokenizer, sampling_params)






