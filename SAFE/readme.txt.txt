-The code in this folder is based on a direct copy of TruthTorchLM. We have altered some functions, and added other new ones. Some of these are clearly marked by comments, but it is not always easy to find exactly what has been changed. In a future project, it might be ideal to comment out the original version and clearly mark the changes.

-The main file to run the experiment is called experiment.py

-The first run in Visual Studo code almost always returns KeyboardInterrupt, but works right after.

-In bm25crossencode.py we use a 18.5GB index folder, which we cannot include in this repository.

-Most of the time, the loop in experiment.py does slowly create new results. However, there have been various errors that will terminate the loop if it runs for long enough. To keep progressing, we added to the seed value and restarted the experiment, providing us the necessary samples.

-In most runs, using qwen2.5-7B is not possible due to the lack of "tools" provided by the endpoint. When faced with this, we swapped to qwen2.5-72B which does not have this issue. It may have been preferable to use a different model, but we felt the 72B version would be closest to our original assignment.

-We have a different evaluate_truth_method_long_form method:
The first difference is in the way the dataset of longform questions is loaded. Rather than download it from a github, we load in a local json file provided by the teaching team of longform queries with get_datasetlocal(). Afterwards, we added BM25+crossencoder results to the context portion of the dataset to achieve our RAG goals.
The second difference is in run_over_(labelled_)dataset, which use a different long_form_generation_with_truth_value. In this function, we introduced batching to combat an error we received regarding multiple calls to a model. 

-We have a different ClaimEvaluator:
call_search in the original ClaimEvaluator used a serper API to google information to verify claims with. We first replaced it with BM25 search to get it working. Afterwards, we tried to use bm25+crossencode for each fact. This did not terminate, so we moved to reuse the 3 topdocs from evaluate_truth_method_long_form.