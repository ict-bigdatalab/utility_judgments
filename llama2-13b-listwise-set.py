import sys
import os
sys.path.append(".")
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["WORLD_SIZE"] = "1"
import transformers
import torch
from tqdm import tqdm
import json
import logging
import argparse
from work.utils.utils import load_source
# from utils.llm import get_llm_result
from work.utils.prompt import get_prompt_multi_docs_all_pair
from vllm import LLM, SamplingParams
def get_args(ra_dict, order, file_type):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='work/dataset/GTI/'+file_type+order+'.json')
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--model_dir', default="llama2-13b-chat")
    parser.add_argument('--type', type=str, default='llama2') ## prompt type
    parser.add_argument('--ra', type=str, default="none", choices=ra_dict.keys())
    parser.add_argument('--outfile', type=str, default='data/method/'+file_type+'-' + order + '.json')
    args = parser.parse_args()

    return args


def main(ra_dict, order, file_type, llm, tokenizer, sampling_params):
    args = get_args(ra_dict, order, file_type)
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
        # exp_useful = "Requirement for utility: \n Utility denotes that employing the passage can help in answering the question.\n"
        # instruct = "Please first provide the answer to the question based on the passage(s) with utility that you have selected and then write 'My selection: [<passage-i>, ...]' in the last line."
        instruct = "Please first provide the answer to the question based on the passage(s) with utility in answering the question that you have selected and then write 'My selection: [<passage-i>, ...]' in the last line."
        exp_useful = ""
        with torch.no_grad():
            for sample in tqdm(all_data[begin:], desc="Filename: %s" % args.outfile):
                passages = sample["passage"]
                question = sample["question"]
                number = len(sample["passage"])
                passages_types = sample["passages_types"]
                labels = sample["ground_truth_label"]
                total = []
                for i, line in enumerate(passages):
                    total.append(("Passage-%d: " % i) + line)
                paras = '\n'.join(total)
                prompts = "Given a question: {question}. Please select the passage(s) with utility in answering the question from the following {number} passages: \n{paras} \n"+exp_useful+"\n"+instruct
                prompts = prompts.format(paras=paras, question=question, number=number)
                message_prompt = "Given a question and select the passage(s) with utility in answering the question from the given passages.\n"+exp_useful+"\n"+instruct
                prompts = "<s>[INST] <<SYS>>"+message_prompt+"<</SYS>>" + prompts+"[/INST]"
                print(prompts)
                res = ""
                outputs = llm.generate([prompts,], sampling_params)
                generated_text = ""
                for output in outputs:
                    generated_text = output.outputs[0].text
                # print(generated_text)
                res = generated_text
                print("model_out:", res)
                print('ground_truth:', labels)
                outfile.write(json.dumps({
                    "passages_types": passages_types,
                    "question": question,
                    "prompt": prompts,
                    "passage": passages,
                    "LLM_output_all": res,
                    "ground_truth_label": labels,
                    "message_prompt": message_prompt
                }) + "\n")
    except Exception as e:
        logging.exception(e)

    finally:
        print(args.outfile, " has output %d line(s)." % num_output)
        outfile.close()

def main_qd(ra_dict, order, file_type, llm, tokenizer, sampling_params):
    args = get_args(ra_dict, order, file_type)
    path = 'work/dataset/list-set-only-answer-order/llama2-13b/'
    args.outfile = path+file_type+ "_" +order + '_qd.json'
    if not os.path.exists(path):
        os.makedirs(path)
    args.ra = ra_dict
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
        # exp_useful = "Requirement for utility: \n Utility denotes that employing the passage can help in answering the question.\n"
        # instruct = "Please first provide the answer to the question based on the passage(s) with utility that you have selected and then write 'My selection: [<passage-i>, ...]' in the last line."
        instruct = "Please first provide the answer to the question based on the passage(s) with utility in answering the question that you have selected and then write 'My selection: [<passage-i>, ...]' in the last line."
        exp_useful = ""
        with torch.no_grad():
            for sample in tqdm(all_data[begin:], desc="Filename: %s" % args.outfile):
                passages = sample["passage"]
                question = sample["question"]
                number = len(sample["passage"])
                passages_types = sample["passages_types"]
                labels = sample["ground_truth_label"]
                total = []
                for i, line in enumerate(passages):
                    total.append(("Passage-%d: " % i) + line)
                paras = '\n'.join(total)
                prompts = "Given a question: {question}. Please select the passage(s) with utility in answering the question from the following {number} passages: \n{paras} \n"+exp_useful+"\n"+instruct
                prompts = prompts.format(paras=paras, question=question, number=number)
                message_prompt = "Given a question and select the passage(s) with utility in answering the question from the given passages.\n"+exp_useful+"\n"+instruct
                prompts = "<s>[INST] <<SYS>>"+message_prompt+"<</SYS>>" + prompts+"[/INST]"
                print(prompts)
                res = ""
                outputs = llm.generate([prompts,], sampling_params)
                generated_text = ""
                for output in outputs:
                    generated_text = output.outputs[0].text
                # print(generated_text)
                res = generated_text
                print("model_out:", res)
                print('ground_truth:', labels)
                outfile.write(json.dumps({
                    "passages_types": passages_types,
                    "question": question,
                    "prompt": prompts,
                    "passage": passages,
                    "LLM_output_all": res,
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



    # main(ra_dict, "first", "HQ", llm, tokenizer, sampling_params)
    # main(ra_dict, "1", "HQ", llm, tokenizer, sampling_params)
    # main(ra_dict, "2", "HQ", llm, tokenizer, sampling_params)
    # main(ra_dict, "3", "HQ", llm, tokenizer, sampling_params)
    # main(ra_dict, "4", "HQ", llm, tokenizer, sampling_params)
    # main(ra_dict, "5", "HQ", llm, tokenizer, sampling_params)
    # main(ra_dict, "6", "HQ", llm, tokenizer, sampling_params)
    # main(ra_dict, "7", "HQ", llm, tokenizer, sampling_params)
    # main(ra_dict, "8", "HQ", llm, tokenizer, sampling_params)
    # main(ra_dict, "9", "HQ", llm, tokenizer, sampling_params)
    # main(ra_dict, "random", "HQ", llm, tokenizer, sampling_params)
    main(ra_dict, "first", "MSM", llm, tokenizer, sampling_params)
    main(ra_dict, "1", "MSM", llm, tokenizer, sampling_params)
    main(ra_dict, "2", "MSM", llm, tokenizer, sampling_params)
    main(ra_dict, "3", "MSM", llm, tokenizer, sampling_params)
    main(ra_dict, "4", "MSM", llm, tokenizer, sampling_params)
    main(ra_dict, "5", "MSM", llm, tokenizer, sampling_params)
    main(ra_dict, "6", "MSM", llm, tokenizer, sampling_params)
    main(ra_dict, "7", "MSM", llm, tokenizer, sampling_params)
    main(ra_dict, "8", "MSM", llm, tokenizer, sampling_params)
    main(ra_dict, "9", "MSM", llm, tokenizer, sampling_params)
    main(ra_dict, "random", "MSM", llm, tokenizer, sampling_params)
    
    # main(ra_dict, "random", "NQ", llm, tokenizer, sampling_params)
    # main(ra_dict, "first", "NQ", llm, tokenizer, sampling_params)
    # main(ra_dict, "1", "NQ", llm, tokenizer, sampling_params)
    # main(ra_dict, "2", "NQ", llm, tokenizer, sampling_params)
    # main(ra_dict, "3", "NQ", llm, tokenizer, sampling_params)
    # main(ra_dict, "4", "NQ", llm, tokenizer, sampling_params)
    # main(ra_dict, "5", "NQ", llm, tokenizer, sampling_params)
    # main(ra_dict, "6", "NQ", llm, tokenizer, sampling_params)
    # main(ra_dict, "7", "NQ", llm, tokenizer, sampling_params)
    # main(ra_dict, "8", "NQ", llm, tokenizer, sampling_params)
    # main(ra_dict, "9", "NQ", llm, tokenizer, sampling_params)


    






    



