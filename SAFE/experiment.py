from sentence_transformers import CrossEncoder
import requests
import bm25s
import json
import pickle
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import TruthTorchLM as ttlm
import torch
import numpy as np



os.environ["OPENROUTER_API_KEY"] = ""
api_model = "openrouter/qwen/qwen-2.5-72b-instruct"

mat_deg_unc = ttlm.truth_methods.MatrixDegreeUncertainty()
ecc_unc = ttlm.truth_methods.EccentricityUncertainty()

truth_methods = [mat_deg_unc, ecc_unc]

chat = [{"role": "system", "content": "You are a helpful assistant. Give short and precise answers."},
        {"role": "user", "content": "What is the capital city of France?"}]



import TruthTorchLM.long_form_generation as LFG
from transformers import DebertaForSequenceClassification, DebertaTokenizer
import longformeval as lfe
import ClaimEvaluator as ce
 
#define a decomposition method that breaks the the long text into claims
decomposition_method = LFG.decomposition_methods.StructuredDecompositionAPI(model=api_model, decomposition_depth=1) #Utilize API models to decompose text

#entailment model is used by some truth methods and claim check methods
model_for_entailment = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli').to('cuda:0')
tokenizer_for_entailment = DebertaTokenizer.from_pretrained('microsoft/deberta-large-mnli')

#define the claim check methods that applies truth methods
qa_generation = LFG.claim_check_methods.QuestionAnswerGeneration(model=api_model, tokenizer=None, num_questions=2, max_answer_trials=2,
                                                                     truth_methods=truth_methods, seed=0,
                                                                     entailment_model=model_for_entailment, entailment_tokenizer=tokenizer_for_entailment) #HF model and tokenizer can also be used, LM is used to generate question
#there are some claim check methods that are directly designed for this purpose, not utilizing truth methods
ac_entailment = LFG.claim_check_methods.AnswerClaimEntailment( model=api_model, tokenizer=None,
                                                                      num_questions=3, num_answers_per_question=2,
                                                                      entailment_model=model_for_entailment, entailment_tokenizer=tokenizer_for_entailment) #HF model and tokenizer can also be used, LM is used to generate question
#generate a message with a truth value, it's a wrapper fucntion for litellm.completion in litellm

#create safe object that assigns labels to the claims
safe = ce.ClaimEvaluator(rater=api_model, tokenizer = None, max_steps = 1, max_retries = 1, num_searches = 1)

#Define metrics
sample_level_eval_metrics = ['f1'] #calculate metric over the claims of a question, then average across all the questions
#dataset_level_eval_metrics = ['auroc', 'prr'] #calculate the metric across all claims
dataset_level_eval_metrics = ['prr'] #calculate the metric across all claims

for see in range(100):
    seed = see+67
    print(f"Running seed {seed}...")

    results = lfe.evaluate_truth_method_long_form(
        dataset='longfact_objects',
        model=api_model,
        tokenizer=None,
        sample_level_eval_metrics=sample_level_eval_metrics,
        dataset_level_eval_metrics=dataset_level_eval_metrics,
        decomp_method=decomposition_method,
        claim_check_methods=[qa_generation],
        claim_evaluator=safe,
        size_of_data=1,
        previous_context=[{'role': 'system', 'content': 'You are a helpful assistant. Give precise answers.'}],
        user_prompt="Question: {question}",
        seed=seed,
        return_method_details=False,
        return_calim_eval_details=False,
        wandb_run=None,
        add_generation_prompt=True,
        continue_final_message=False
    )

    # Choose your filename style: result0.txt, result1.txt, ...
    filename = f"result{seed}.txt"
    print(filename)
    print("\n")
    print(results) 
    print("\n")
    # Write output to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(str(results))

    unc_methods = np.array(results['output_dict']['claim_check_methods_0']['normalized_truth_values'])
    np.save(f'unc_methods{seed}.npy', unc_methods)
    safe_scores = []
    for key in results['sample_level_eval_list'].keys():
        safe_scores.append(results['sample_level_eval_list'][key]['f1']['mean'])
    safe_f1_means = np.array(safe_scores)
    np.save(f"safe{seed}.npy", safe_f1_means)


    print(f"Saved results to {filename}\n")

