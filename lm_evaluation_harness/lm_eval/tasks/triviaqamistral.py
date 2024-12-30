"""
TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension
https://arxiv.org/pdf/1705.03551.pdf

TriviaQA is a reading comprehension dataset containing over 650K question-answer-evidence
triples. TriviaQA includes 95K question-answer pairs authored by trivia enthusiasts
and independently gathered evidence documents, six per question on average, that provide
high quality distant supervision for answering the questions.

Homepage: https://nlp.cs.washington.edu/triviaqa/
"""
import inspect
import lm_eval.datasets.triviaqa.triviaqa
from lm_eval.utils import create_dataloader
from lm_eval.base import Task, rf
from lm_eval.metrics import mean, weighted_perplexity
from lm_eval import utils
import datasets
import random


_CITATION = """
@InProceedings{JoshiTriviaQA2017,
    author = {Joshi, Mandar and Choi, Eunsol and Weld, Daniel S. and Zettlemoyer, Luke},
    title = {TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension},
    booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics},
    month = {July},
    year = {2017},
    address = {Vancouver, Canada},
    publisher = {Association for Computational Linguistics},
}
"""


"""
TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension
https://arxiv.org/pdf/1705.03551.pdf

TriviaQA is a reading comprehension dataset containing over 650K question-answer-evidence
triples. TriviaQA includes 95K question-answer pairs authored by trivia enthusiasts
and independently gathered evidence documents, six per question on average, that provide
high quality distant supervision for answering the questions.

Homepage: https://nlp.cs.washington.edu/triviaqa/
"""
import inspect
import lm_eval.datasets.triviaqa.triviaqa
from lm_eval.utils import create_dataloader
from lm_eval.base import Task, rf
from lm_eval.metrics import mean, weighted_perplexity
from lm_eval import utils
import datasets
import random


_CITATION = """
@InProceedings{JoshiTriviaQA2017,
    author = {Joshi, Mandar and Choi, Eunsol and Weld, Daniel S. and Zettlemoyer, Luke},
    title = {TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension},
    booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics},
    month = {July},
    year = {2017},
    address = {Vancouver, Canada},
    publisher = {Association for Computational Linguistics},
}
"""

import os
import glob

class TRIVIAQAMISTRAL():
    VERSION = 0
    DATASET_PATH = "triviaqamistral"
    DATASET_NAME = None

    def __init__(self):

        self.dataset = datasets.load_dataset(
            "json",
            data_files={'train':['lm_eval/datasets/triviaqa/triviaqa-mistral.json']} # ,'test': ''}
        )
        self._training_docs = None
        self._validation_docs = None
        self._fewshot_docs = None
        self._training_docs = None
        self._validation_docs = None
        self._fewshot_docs = None
        

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs
    
    def validation_docs(self):
        raise NotImplementedError

    def test_docs(self):
        return self.dataset["test"]

    
    def get_dataloader(self, tokenizer, split = 'train', subset_size = None, batch_size = 1, num_fewshot = 0):
        docs = self.training_docs() if split == 'train' else self.test_docs() if split == 'test' else RuntimeError(f'Data not available for {split} split.')
        return create_dataloader(tokenizer, docs, self.fewshot_context, self.doc_to_cont, subset_size = subset_size, batch_size = batch_size, num_fewshot = num_fewshot)
    
    def doc_to_cont(self, doc):
        return doc['output'] if "output" in doc.keys() else doc["expected_output"]

    def doc_to_text(self, doc):
        return doc["instruction"]+ doc["input"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc['output'] if "output" in doc.keys() else doc["expected_output"]
    
    def fewshot_examples(self, k, rnd):
        if self._training_docs is None:
            self._training_docs = list(self.training_docs())

        return rnd.sample(self._training_docs, k)
    
    # @classmethod
    # def partial_context(cls, doc, option):
    #     # Substitute the pronoun in the sentence with the specified option
    #     # and ignore everything after.
    #     pronoun_loc = doc["sentence"].index("_")
    #     return doc["sentence"][:pronoun_loc] + option
    
    # @classmethod
    # def partial_target(cls, doc):
    #     # The target is everything after the document specified pronoun.
    #     pronoun_loc = doc["sentence"].index("_") + 1
    #     return " " + doc["sentence"][pronoun_loc:].strip()
    
    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        target = doc['output'] if "output" in doc.keys() else doc["expected_output"] # doc
        req = rf.loglikelihood(self.doc_to_text(doc), target)
        return req

    
    @classmethod
    def count_words(cls, doc):
        import re
        """Downstream tasks with custom word boundaries should override this!"""
        return len(doc)

    
    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        # print("results", results)
        (loglikelihood,) = results
        words = self.count_words(doc["output"])
        completion = results[0][1]
        # answer = doc["output"]
        return {
            "word_perplexity": (loglikelihood[0], words),
            "acc": completion
        }
    
    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {
            "word_perplexity": weighted_perplexity,
            "acc": mean
        }
    
    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            "word_perplexity": True,
            "acc": True
        }
    
    @utils.positional_deprecated
    def fewshot_context(
        self, doc, num_fewshot, provide_description=None, rnd=random, description=None
    ):
        """Returns a fewshot context string that is made up of a prepended description
        (if provided), the `num_fewshot` number of examples, and an appended prompt example.

        :param doc: str
            The document as returned from training_docs, validation_docs, or test_docs.
        :param num_fewshot: int
            The number of fewshot examples to provide in the returned context string.
        :param provide_description: bool
            Not implemented, and this option is deprecated and will be removed in a future version in favor of a different description providing method
        :param rnd: random.Random
            The pseudo-random number generator used to randomly sample examples.
            WARNING: This is currently a required arg although it's optionalized with a default `None`.
        :param description: str
            The task's description that will be prepended to the fewshot examples.
        :returns: str
            The fewshot context.
        """

        assert (
            rnd is not None
        ), "A `random.Random` generator argument must be provided to `rnd`"
        assert not provide_description, (
            "The `provide_description` arg will be removed in future versions. To prepend "
            "a custom description to the context, supply the corresponding string via the "
            "`description` arg."
        )
        if provide_description is not None:
            # nudge people to not specify it at all
            print(
                "WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict"
            )

        description = description + "\n\n" if description else ""

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            # for sets with no training docs, draw from other set *but ensure no overlap with current doc*
            if self.has_training_docs():
                fewshotex = self.fewshot_examples(k=num_fewshot, rnd=rnd)
            else:
                if self._fewshot_docs is None:
                    self._fewshot_docs = list(
                        self.validation_docs()
                        if self.has_validation_docs()
                        else self.test_docs()
                    )

                fewshotex = rnd.sample(self._fewshot_docs, num_fewshot + 1)

                # get rid of the doc that's the one we're evaluating, if it's in the fewshot
                fewshotex = [x for x in fewshotex if x != doc][:num_fewshot]

            labeled_examples = (
                "\n\n".join(
                    [
                        self.doc_to_text(doc) + self.doc_to_target(doc)
                        for doc in fewshotex
                    ]
                )
                + "\n\n"
            )

        example = self.doc_to_text(doc)
        return description + labeled_examples + example
    
