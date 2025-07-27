import re
import textwrap
from typing import List

from PIL import Image

from ..base import BaseVLLMAgent



class ReasoningAnswerableChecker(object):

    def __init__(self):
        black_semi_negative_words = ['sorry', 'insufficient', 'incomplete']
        black_negative_words = ['know', 'sure', 'determine', 'identif', 'clear', 'provide', 'confident', 'sufficient', 'have access', 'have enough information', 'have information', 'answer', 'specif']
        self.reasoning_black_list = \
            [f"n't {word}" for word in black_negative_words] + [f"not {word}" for word in black_negative_words] + \
            [f"n't possible to {word}" for word in black_negative_words] + [f"not possible to {word}" for word in black_negative_words] + \
            [f"impossible to {word}" for word in black_negative_words] + black_semi_negative_words
        self.need_rag_re_patterns = [r'(?:\d{3,}(?:\.\d+)?|\d*\.\d+)']
        self.query_black_list = ['engine', 'animal', ' name', 'described', ' type', ' food', 'typical', ' range', ' generations', ' soil', 'toyota', 'height', ' base ', ' origin', ' u.s.', ' family', ' miles', ' suv', ' calories', ' belongs', ' built', ' gram', ' sculpture', ' kind ', ' price', ' designed', ' versus', ' available']

    def __call__(self, queries: List[str], reasonings: List[str]) -> List[bool]:
        lower_queries = [q.lower() for q in queries]
        lower_reasonings = [r.lower() for r in reasonings]
        # TODO: `need rag` is not an answerable.
        # "i don't know" is treated as answerable ("cannot_answer": false)
        answerable = [False] * len(queries)
        answerable = [False if re.search(pattern, query) else ans for pattern in self.need_rag_re_patterns for ans, query in zip(answerable, lower_queries)]
        answerable = [False if any(keyword in query for keyword in self.query_black_list) else ans for ans, query in zip(answerable, lower_queries)]
        answerable = [True if (reasoning.startswith("i don't know") and len(reasoning) <= 15) or not any(keyword in reasoning for keyword in self.reasoning_black_list) else ans for ans, reasoning in zip(answerable, lower_reasonings)]
        return answerable


class NeedTextVerifyChecker(object):
    
    def __init__(self, all=False):
        self.need_text_verify_query_list = [' sale ', ' sales ', ' price ', ' prices ', ' cost ', ' costs ', ' times ', ' year ', ' years ', ' month ', ' months ', ' day ', ' days ', ' gram ', ' grams '] + ['typical', 'different', 'leaves', 'grow', 'orchid', 'usually', 'status', 'served', 'structure', 'conservation', 'constructed', 'long', 'benefit', 'diet', 'period', 'color', 'restaurant', 'differ', 'offer', 'capacity', 'fuel', 'countries', 'animals', 'health', 'area', 'thrive', 'time', 'show', 'common', 'largest', 'church', 'president', ' us ', ' us?', 'floor', 'cold', 'weight', 'number', 'species', 'common', 'much', 'plant', 'tree', 'region', 'compare', 'flower', ' get ', 'model']
        self.need_text_verify_reasoning_list = [' year ', ' years ', ' month ', ' months ', ' day ', ' days ', ' gram', ' grams']
        # Pattern to match queries or reasonings containing more than one integer or decimal number
        self.need_text_verify_re_patterns = [r'(?:\d{3,}(?:\.\d+)?|\d*\.\d+)']
        self.all = all
        
    def __call__(self, queries: List[str], reasonings: List[str]) -> List[bool]:
        if self.all:
            return [True] * len(queries)
        queries_lower = [query.lower() for query in queries]
        reasonings_lower = [reasoning.lower() for reasoning in reasonings]
        return [
            any(keyword in query_lower for keyword in self.need_text_verify_query_list)
            or any(keyword in reasoning_lower for keyword in self.need_text_verify_reasoning_list)
            or re.search(pattern, query_lower)
            or re.search(pattern, reasoning_lower)
            for pattern in self.need_text_verify_re_patterns
            for query_lower, reasoning_lower in zip(queries_lower, reasonings_lower, strict=True)
        ]


class DirectAnswerQueryChecker(object):

    def __init__(self):
        self.white_query_list = ['translate', 'meaning', 'word', 'english', 'language', 'diagram', 'illustrate', 'formula', 'equation']
        
    def __call__(self, queries: List[str], reasonings: List[str]) -> List[bool]:
        queries_lower = [query.lower() for query in queries]
        return [True if any(keyword in query_lower for keyword in self.white_query_list) else False for query_lower in queries_lower]


class RuleAnswerableRouter(object):

    def __init__(self, all_text_verify=False):
        self.direct_answer_checker = DirectAnswerQueryChecker()
        self.answerable_checker = ReasoningAnswerableChecker()
        self.need_text_verify_checker = NeedTextVerifyChecker(all=all_text_verify)

    def __call__(self, images: List[Image.Image], queries: List[str], reasonings: List[str], domains: List[str]) -> List[str]:
        direct_answer_flag = self.direct_answer_checker(queries=queries, reasonings=reasonings)
        answerable_flag = self.answerable_checker(queries=queries, reasonings=reasonings)
        text_verify_flag = self.need_text_verify_checker(queries=queries, reasonings=reasonings)
        output_flag = []
        indices = dict(idk = [], need_rag = [], text_verify = [], direct_output = [])
        for i, (a, d, t, domain) in enumerate(zip(answerable_flag, direct_answer_flag, text_verify_flag, domains, strict=True)):
            # set as idk if direct output but not answerable
            if (not a and d) or reasonings[i].lower().startswith("i don't know"):
                output_flag.append('idk')
                indices['idk'].append(i)
            # set as need rag if not answerable and not direct output
            elif domain == 'text or chart reasoning':
                output_flag.append('direct_output')
                indices['direct_output'].append(i)
            elif not a:
                output_flag.append('need_rag')
                indices['need_rag'].append(i)
            # set as direct output if direct output
            elif d:
                output_flag.append('direct_output')
                indices['direct_output'].append(i)
            # set as text verify if need text verify, and no need rag and direct output
            elif t:
                output_flag.append('text_verify')
                indices['text_verify'].append(i)
            # set as direct output if no need text verify and no need rag
            else:
                output_flag.append('direct_output')
                indices['direct_output'].append(i)
        return output_flag, indices
    
    
class AnswerableRouter(BaseVLLMAgent):
    
    def __init__(self, llm, tokenizer, generation_config, formater):
        print(("DeprecationWarning: This class is deprecated. Use RuleAnswerableRouter instead."))
        super().__init__(llm, tokenizer, generation_config, formater)
        black__semi_negative_words = ['sorry', 'insufficient']
        black_negative_words = ['know', 'sure', 'determine', 'identif', 'clear', 'provide', 'confident', 'sufficient', 'have access']
        self.black_list = [f"n't {word}" for word in black_negative_words] + [f"not {word}" for word in black_negative_words] + \
                          [f"n't possible to {word}" for word in black_negative_words] + [f"not possible to {word}" for word in black_negative_words] + \
                          [f"impossible to {word}" for word in black_negative_words] + black__semi_negative_words
            
    
    @property 
    def _purpose(self):
        return "determine whether the given reasoning text can answer the query."
    
    @property
    def _schema_key(self):
        return "cannot_answer"
    
    @property
    def _value_type(self):
        return '<true/false>'
    
    @property
    def _system_prompt(self):
        return textwrap.dedent("""
                You are a verifier to determine whether the user's step-wise reasoning can answer the query. 
                The reasoning should be considered unanswerable if it meets any of the following conditions:
                    1. The reasoning explicitly states that it cannot answer the query, identify the object, or interpret the image due to insufficient information or lack of relevant knowledge.
                    2. The reasoning fails to refer to a specific object or entity relevant to the query, instead using only vague or generic terms (e.g., "the car", "the animal", "the mountain") rather than precise identifiers (e.g., "the Toyota Camry", "the giraffe", "Mount Fuji").
                    3. The reasoning does not provide a direct answer but only suggests possibilities, guesses, or indirect clues (e.g., "the user can check...", "it may depend on...", "perhaps it is...").
                Exception: Reasoning that involves math calculations should always be considered answerable, even if it lacks specific object names.
Output the answerability verification in the following JSON format:
                {"cannot_answer": <true/false>}
                No other text or explanation is allowed.
                """
            ).strip()
        
    def _prompt(self, query: str, reasoning: str):
        user_prompt = textwrap.dedent(f"""
                Given the query text: {query}
                and step-wise reasoning: "{reasoning}"
                Have the given reasoning text answered the query or not?
                """
            ).strip()
        prompts = [
            {
                "role": "system",
                "content": self._system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            },
        ]
        return prompts
    
    def _output_postprocess(self, outputs: List[bool]) -> List[bool]:
        cannot_answer = list()
        for output in outputs:
            if isinstance(output, bool):
                cannot_answer.append(output)
            else:
                try:
                    if isinstance(output, str):
                        val = output.strip().lower()
                        if val == "true" or val == "none":
                            cannot_answer.append(True)
                        elif val == "false":
                            cannot_answer.append(False)
                        else:
                            raise ValueError(f"Cannot convert string to bool: {output}")
                    else:
                        cannot_answer.append(bool(output))
                except Exception as e:
                    print(f"[TempAnswerableRouter] Exception in _output_postprocess: {e}. Setting to False.")
                    cannot_answer.append(False)
        return cannot_answer
    
    def __call__(self, queries: List[str], reasonings: List[str]) -> List[bool]:
        """
        Output decision logic for answerable router:
        - True in the output means the query is considered NOT answerable (i.e., "cannot_answer": true).
            - This happens if the reasoning is in the black list (contains any black list keyword), or for any other case where the model determines the answer is not possible.
        - False in the output means the query is considered answerable (i.e., "cannot_answer": false).
            - This happens if the reasoning is not in the black list and does not start with "i don't know".
        - Special case: If the reasoning starts with "i don't know", it is treated as answerable (False/"cannot_answer": false), so the system can directly output this response.
        """
        lower_reasonings = [r.lower() for r in reasonings]
        # "i don't know" is treated as answerable (False/"cannot_answer": false)
        non_black_index = [
            i for i, r in enumerate(lower_reasonings)
            if r.startswith("i don't know") or not any(keyword in r for keyword in self.black_list)
        ]
        final_cannot_answer = [True] * len(queries)  # Default: cannot answer (True)
        finnal_raw_cannot_answer = ['hit black list'] * len(queries)
        non_black_queries = [queries[i] for i in non_black_index]
        non_black_reasonings = [reasonings[i] for i in non_black_index]
        cannot_answer, raw_cannot_answer = super().__call__(query=non_black_queries, reasoning=non_black_reasonings)
        for i, a, r in zip(non_black_index, cannot_answer, raw_cannot_answer):
            final_cannot_answer[i] = a
            finnal_raw_cannot_answer[i] = r
        return final_cannot_answer, finnal_raw_cannot_answer