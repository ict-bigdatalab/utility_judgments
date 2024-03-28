import sys
import os
sys.path.append(".")
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
os.environ["WORLD_SIZE"] = "1"
import transformers
import torch
from tqdm import tqdm
import json
import logging
import argparse
from work.utils.utils import load_source
from vllm import LLM, SamplingParams
def get_args(ra_dict, order, file_type):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="work/dataset/GTI/"+file_type+order+".json")
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--model_dir', default="llama2-13b-chat")
    parser.add_argument('--type', type=str, default='llama2') ## prompt type
    parser.add_argument('--ra', type=str, default="none", choices=ra_dict.keys())
    parser.add_argument('--outfile', type=str, default='data/method/'+file_type+'-' + order + '.json')
    args = parser.parse_args()
    path = 'work/dataset/list-rank-real-only-answerijk/llama13b/'
    args.outfile = path+file_type+ "_" +order + '.json'
    if not os.path.exists(path):
        os.makedirs(path)
    args.ra = ra_dict

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
    print("begin:", begin)
    
    try:
        exp_useful = ""
        instruct = "Please first provide the answer based on the passages that you have ranked in utility and then  write the ranked passages in descending order of utility in answering the question, like 'My rank: [i]>[j]>...>[k]'."
        with torch.no_grad():
            re_all_data = all_data[begin:]
            for i in tqdm(range(0, len(re_all_data), 10), desc="Filename: %s" % args.outfile):
                samples = re_all_data[i: min(i+10, len(re_all_data))]
                prompt_list = []
                passagess = []
                questions = []
                passages_typeses = []
                labelses = []
                for sample in samples:
                    passages = sample["passage"]
                    passagess.append(passages)
                    question = sample["question"]
                    questions.append(question)
                    number = len(sample["passage"])
                    passages_types = sample["passages_types"]
                    passages_typeses.append(passages_types)
                    labels = sample["ground_truth_label"]
                    labelses.append(labels)
                    total = []
                    for i, line in enumerate(passages):
                        total.append(("[%d]: " % i) + line)
                    paras = '\n'.join(total)
                    prompts = "Given a question: {question}. Please rank all  following {number} passages based on their utility in answering the question: \n{paras} \n" +exp_useful+"\n"+instruct
                    prompts = prompts.format(paras=paras, question=question, number=number)
                    message_prompt = "Given a question and rank all following passages in descending order based on their utility in answering the question.\n"
                    prompts = "<s>[INST] <<SYS>>"+message_prompt+"<</SYS>>" + prompts+"[/INST]"
                    prompt_list.append(prompts)
                outputs = llm.generate(prompt_list, sampling_params)
                if len(outputs) != len(questions):
                    outputs = []
                    for prompts in prompt_list:
                        output = llm.generate([prompts], sampling_params)
                        outputs += [output[0]]
                    
                assert len(outputs) == len(questions)
                for index, output in enumerate(outputs):
                    prompt = output.prompt # 获取原始的输入提示
                    generated_text = output.outputs[0].text
                    res = generated_text
                    print("prompts: ", prompt)
                    print("model_out:", res)
                    print('ground_truth:', labels)
                    outfile.write(json.dumps({
                        "passages_types": passages_typeses[index],
                        "question": questions[index],
                        "prompt": prompt,
                        "passage": passagess[index],
                        "LLM_output_all": res,
                        "ground_truth_label": labelses[index]
                    }) + "\n")
    except Exception as e:
        logging.exception(e)

    finally:
        # print(args.outfile, " has output %d line(s)." % num_output)
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
   


