from transformers import pipeline
import torch

import requests



def get_formality_llama(txt, hf_token):
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    prompt = """
    You are an expert linguist trained to assess the formality of sentences. Your task is to evaluate the formality in a given sentence, 
    where the value "YES" indicates extremely formal language, such as what might be used in official documents or academic papers, 
    and the value "NO" indicates informal language, typical of everyday communication or slang, casual text messages, or colloquial expressions.
    Consider factors such as vocabulary choice, sentence structure, and tone when assessing the formality.

    Examples:
    1) The formal sentence "Our financial requirements are dramatically reduced, and I can afford to offer 
    to bring my experience, energy and talents to helping this administration implement the many changes necessary to carry out it goals which I so strongly support." should be assigned the value "YES".
    2) The formal sentence "Council chair Sir Michael Rawlins says he is stepping down and Nutt succeeds him in October." should be assigned the value "YES".
    3) The formal sentence "The plans come amid growing concern about the number and severity of infections in children." should be assigned the value "YES".
    4) The formal sentence "Both man have many years of experience in international competition." should be assigned the value of "YES".
    5) The informal sentence "so i guess we will see." should be assigned the value "NO".
    6) The informal sentence "that's out there." should be assigned the value "NO".
    7)The informal sentence "Have you ever gotten that little stringy thing from a banana in your mouth." should be assigned the value "NO".
    8) The informal sentence "if theyre really just friends you should try and trust him" should be assigned the value "NO".

    Write only "YES" if the sentence is formal or write only "NO" if it is informal.
    """

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Assign YES or NO to this sentence: {0}".format(txt)},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=4,
        temperature=0.5,
    )

    final_score = outputs[0]["generated_text"][-1]["content"]
    final_score_dict = {"YES": 1, "NO": 0}
    return final_score_dict[final_score]
