"""Code taken from SAFE https://github.com/google-deepmind/long-form-factuality/tree/main/eval/safe and adapted to TruthTorchLM"""

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rates a single atomic fact for accuracy."""

import dataclasses
import re
from typing import Union, Any
from litellm import completion
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
import TruthTorchLM.long_form_generation.utils.safe_utils as utils
from abc import ABC
from TruthTorchLM.utils.common_utils import generate, fix_tokenizer_chat



SUPPORTED_LABEL = "Supported"
NOT_SUPPORTED_LABEL = "Not Supported"

_STATEMENT_PLACEHOLDER = "[STATEMENT]"
_KNOWLEDGE_PLACEHOLDER = "[KNOWLEDGE]"
_NEXT_SEARCH_FORMAT = f"""\
Instructions:
1. You have been given a STATEMENT and some KNOWLEDGE points.
2. Your goal is to try to find evidence that either supports or does not \
support the factual accuracy of the given STATEMENT.
3. To do this, you are allowed to issue ONE Google Search query that you think \
will allow you to find additional useful evidence.
4. Your query should aim to obtain new information that does not appear in the \
KNOWLEDGE. This new information should be useful for determining the factual \
accuracy of the given STATEMENT.
5. Format your final query by putting it in a markdown code block.

KNOWLEDGE:
{_KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}
"""
_FINAL_ANSWER_FORMAT = f"""\
Instructions:
1. You have been given a STATEMENT and some KNOWLEDGE points.
2. Determine whether the given STATEMENT is supported by the given KNOWLEDGE. \
The STATEMENT does not need to be explicitly supported by the KNOWLEDGE, but \
should be strongly implied by the KNOWLEDGE.
3. Before showing your answer, think step-by-step and show your specific \
reasoning. As part of your reasoning, summarize the main points of the \
KNOWLEDGE.
4. If the STATEMENT is supported by the KNOWLEDGE, be sure to show the \
supporting evidence.
5. After stating your reasoning, restate the STATEMENT and then determine your \
final answer based on your reasoning and the STATEMENT.
6. Your final answer should be either "{SUPPORTED_LABEL}" or \
"{NOT_SUPPORTED_LABEL}". Wrap your final answer in square brackets.

KNOWLEDGE:
{_KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}
"""


class ClaimEvaluator(ABC):

    def __init__(
        self,
        rater: Union[PreTrainedModel, str],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
        max_steps: int = 1,
        max_retries: int = 3,
        num_searches: int = 1,
    ):
        self.rater = rater
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.num_searches = num_searches

    def __call__(self, atomic_fact: str, knowledge_docs: str = "") -> dict:
        return check_atomic_fact(
            atomic_fact=atomic_fact,
            rater=self.rater,
            tokenizer=self.tokenizer,
            max_steps=self.max_steps,
            max_retries=self.max_retries,
            num_searches=self.num_searches,
            knowledge_docs=knowledge_docs, #new context top3 docs!
        )



@dataclasses.dataclass()
class GoogleSearchResult:
    query: str
    result: str


@dataclasses.dataclass()
class FinalAnswer:
    response: str
    answer: str


def _generate(prompt, model, tokenizer, **kwargs):
    messages = [{"role": "user", "content": prompt}]
    if type(model) == str:
        response = completion(model=model, messages=messages, **kwargs)
        generated_text = response.choices[0].message["content"]
    else:
        tokenizer, messages = fix_tokenizer_chat(tokenizer, messages)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            continue_final_message=False,
        )
        generated_text = generate(text, model, tokenizer, **kwargs)[
            "generated_text_skip_specials"
        ]
    return generated_text



def call_search(str):
    "Removed for local implementation"
    return




def maybe_get_next_search(
    atomic_fact: str,
    past_searches: list[GoogleSearchResult],
    model: Union[PreTrainedModel, str],
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
    num_searches: int = 1,
    knowledge_docs="",
    **kwargs,
) -> Union[GoogleSearchResult, None]:
    """Get the next query from the model."""
    knowledge = "\n".join([s.result for s in past_searches])
    knowledge = "N/A" if not knowledge else knowledge
    full_prompt = _NEXT_SEARCH_FORMAT.replace(
        _STATEMENT_PLACEHOLDER, atomic_fact)
    full_prompt = full_prompt.replace(_KNOWLEDGE_PLACEHOLDER, knowledge)
    full_prompt = utils.strip_string(full_prompt)
    model_response = _generate(full_prompt, model, tokenizer, **kwargs)
    query = utils.extract_first_code_block(
        model_response, ignore_language=True)
    print(f'Search query: {query}')

    if model_response and query:
        return GoogleSearchResult(
            query=query,
            result=knowledge_docs if knowledge_docs else "No knowledge available."
        )

    return None


def maybe_get_final_answer(
    atomic_fact: str,
    searches: list[GoogleSearchResult],
    model: Union[PreTrainedModel, str],
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
    knowledge_docs="",
    **kwargs,
) -> Union[FinalAnswer, None]:
    """Get the final answer from the model."""
    knowledge = knowledge_docs
    full_prompt = _FINAL_ANSWER_FORMAT.replace(
        _STATEMENT_PLACEHOLDER, atomic_fact)
    full_prompt = full_prompt.replace(_KNOWLEDGE_PLACEHOLDER, knowledge)
    full_prompt = utils.strip_string(full_prompt)
    model_response = _generate(full_prompt, model, tokenizer, **kwargs)
    answer = utils.extract_first_square_brackets(model_response)
    answer = re.sub(r"[^\w\s]", "", answer).strip()

    if model_response and answer in [SUPPORTED_LABEL, NOT_SUPPORTED_LABEL]:
        return FinalAnswer(response=model_response, answer=answer)

    return None

#Changed to use knowledge_docs, which is the context provided by RAG
def check_atomic_fact(
    atomic_fact: str,
    rater: Union[PreTrainedModel, str],
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
    max_steps: int = 1,
    max_retries: int = 3,
    num_searches: int = 1,
    knowledge_docs: str = "",
    **kwargs,
) -> tuple[Union[FinalAnswer, None], dict[str, Any]]:
    """Check if the given atomic fact is supported."""
    search_results = []

    for i in range(max_steps):
        next_search, num_tries = None, 0

        while not next_search and num_tries <= max_retries:
            # print(f'Step {i} Search trial #{num_tries}')s
            next_search = maybe_get_next_search(
            atomic_fact=atomic_fact,
            past_searches=search_results,
            model=rater,
            tokenizer=tokenizer,
            num_searches=num_searches,
            knowledge_docs=knowledge_docs,
            **kwargs,
        )

            num_tries += 1

        if next_search is None:
            utils.maybe_print_error("Unsuccessful parsing for `next_search`")
            break
        else:
            search_results.append(next_search)

    search_dicts = {"google_searches": [
        dataclasses.asdict(s) for s in search_results]}
    final_answer, num_tries = None, 0

    while not final_answer and num_tries <= max_retries:
        num_tries += 1
        final_answer = maybe_get_final_answer(
            atomic_fact=atomic_fact,
            searches=search_results,
            model=rater,
            tokenizer=tokenizer,
            knowledge_docs=knowledge_docs,
            **kwargs,
        )


    if final_answer is None:
        utils.maybe_print_error("Unsuccessful parsing for `final_answer`")
        return {"answer": None, "response": None, "search_details": None}

    return {
        "answer": final_answer.answer,
        "response": final_answer.response,
        "search_details": search_dicts,
    }