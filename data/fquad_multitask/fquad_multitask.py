"""TODO(fquad): Add a description here."""

from __future__ import absolute_import, division, print_function

import json
import os
import logging

import nltk
nltk.download('punkt')

import nlp


# TODO(fquad): BibTeX citation
_CITATION = """\
@ARTICLE{2020arXiv200206071
       author = {{Martin}, d'Hoffschmidt and {Maxime}, Vidal and
         {Wacim}, Belblidia and {Tom}, Brendlé},
        title = "{FQuAD: French Question Answering Dataset}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computation and Language},
         year = "2020",
        month = "Feb",
          eid = {arXiv:2002.06071},
        pages = {arXiv:2002.06071},
archivePrefix = {arXiv},
       eprint = {2002.06071},
 primaryClass = {cs.CL}
}
"""

# TODO(fquad):
_DESCRIPTION = """\
FQuAD: French Question Answering Dataset
We introduce FQuAD, a native French Question Answering Dataset. FQuAD contains 25,000+ question and answer pairs.
Finetuning CamemBERT on FQuAD yields a F1 score of 88% and an exact match of 77.9%.

"""
_URL = "https://storage.googleapis.com/illuin/fquad"
_TRAIN_DATA = "train.json.zip"
_VALID_DATA = "valid.json.zip"

QG_FORMATS = [
    "prepend",
    "highlight",
    "prepend_highlight",
]

class FquadMultitaskConfig(nlp.BuilderConfig):
    """BuilderConfig for SQUAD."""

    def __init__(self, qg_format="highlight", **kwargs):
        """BuilderConfig for SQUAD.

    Args:
      **kwargs: keyword arguments forwarded to super.
    """
        super(FquadMultitaskConfig, self).__init__(**kwargs)
        self.qg_format = qg_format


class FquadMultitask(nlp.GeneratorBasedBuilder):
    """TODO(fquad): Short description of my dataset."""

    # TODO(fquad): Set up version.
    VERSION = nlp.Version("0.1.0")

    BUILDER_CONFIGS = [
        FquadMultitaskConfig(
            name=f"{format_}_qg_format",
            version=nlp.Version("1.0.0", "New split API (https://tensorflow.org/datasets/splits)"),
            description="Plain text",
            qg_format=format_
        )
        for format_ in QG_FORMATS
    ]

    def _info(self):
        return nlp.DatasetInfo(
            description=_DESCRIPTION,
            features=nlp.Features(
                {
                    "source_text": nlp.Value("string"),
                    "target_text": nlp.Value("string"),
                    "task": nlp.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://fquad.illuin.tech/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(fquad): Downloads the data and defines the splits
        # dl_manager is a nlp.download.DownloadManager that can be used to
        # download and extract URLs
        download_urls = {"train": os.path.join(_URL, _TRAIN_DATA), "valid": os.path.join(_URL, _VALID_DATA)}
        dl_dir = dl_manager.download_and_extract(download_urls)
        return [
            nlp.SplitGenerator(
                name=nlp.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(dl_dir["train"], "train.json")},
            ),
            nlp.SplitGenerator(
                name=nlp.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(dl_dir["valid"], "valid.json")},
            ),
        ]

    def _get_correct_alignement(self, context, answer):
        """ Some original examples in SQuAD have indices wrong by 1 or 2 character. We test and fix this here. """
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)
        if context[start_idx:end_idx] == gold_text:
            return start_idx, end_idx       # When the gold label position is good
        elif context[start_idx-1:end_idx-1] == gold_text:
            return start_idx-1, end_idx-1   # When the gold label is off by one character
        elif context[start_idx-2:end_idx-2] == gold_text:
            return start_idx-2, end_idx-2   # When the gold label is off by two character
        else:
            raise ValueError()
    
    def process_qa_text(self, context, question, answer):
        ans_gen_input = f"question: {question}  context: {context}"
        ans_gen_target = f"{answer}"
        return {"source_text": ans_gen_input, "target_text": ans_gen_target, "task": "qa"}

    def process_qg_text(self, context, question, answer):
        answer_text = answer['text'].strip()
        
        if self.config.qg_format == "prepend":
            que_gen_input = f"answer: {answer_text}  context: {context}"
        elif self.config.qg_format == "highlight":
            # start_pos, end_pos = self._get_correct_alignement(context, answer)
            start_pos, end_pos = answer["answer_start"], answer["answer_start"] + len(answer["text"]) 
            que_gen_input = f"generate question: {context[:start_pos]} {{hl_token}} {answer_text} {{hl_token}} {context[end_pos:]}"
        else:
            # start_pos, end_pos = self._get_correct_alignement(context, answer)
            start_pos, end_pos = answer["answer_start"], answer["answer_start"] + len(answer["text"]) 
            que_gen_input = f"answer: {answer_text} context: {context[:start_pos]} {{hl_token}} {answer_text} {{hl_token}} {context[end_pos:]}"
        
        que_gen_target = f"{question}"
        return {"source_text": que_gen_input, "target_text": que_gen_target, "task": "qg"}
    
    def process_e2e_qg(self, paragraph):
        source_text = f"generate questions: {paragraph['context'].strip()}"
        questions = [qas['question'].strip() for qas in paragraph['qas']]
        target_text = " {sep_token} ".join(questions)
        target_text = f"{target_text} {{sep_token}}"
        return {"source_text": source_text, "target_text": target_text, "task": "e2e_qg"}

    def process_ans_ext(self, paragraph):
        context = paragraph['context'].strip()
    
        # split into sentences
        sents = nltk.sent_tokenize(context)

        # get positions of the sentences
        positions = []
        for i, sent in enumerate(sents):
            if i == 0:
                start, end = 0, len(sent)
            else:
                start, end = (prev_end + 1), (prev_end + len(sent) + 1)
            prev_end = end
            positions.append({'start': start, 'end': end})
        
        # get answers
        answers = [qa['answers'][0] for qa in paragraph['qas']]

        # get list of answers for each sentence
        sent_answers = []
        for pos, sent in zip(positions, sents):
            target_answers = []
            for ans in answers:
                if ans['answer_start'] in range(pos['start'], pos['end']):
                    target_answers.append(ans['text'].strip())
            sent_answers.append(target_answers)

        # build inputs and targets
        examples = []
        for i, ans in enumerate(sent_answers):
            context = "extract answers:"
            if len(ans) == 0: continue
            ans = list(set(ans))
            for j, sent in enumerate(sents):
                if i == j:
                    sent = "{hl_token} %s {hl_token}" % sent
                context = "%s %s" % (context, sent)
                context = context.strip()
            input_text = context
            target_text = " {sep_token} ".join(ans) + " {sep_token}"

            examples.append({'source_text': input_text, "target_text": target_text, "task": "ans_ext"})
        
        return examples

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logging.info("generating examples from = %s", filepath)
        count = 0
        tasks = ['qa', 'qg', 'ans_ext', 'e2e_qg']
        with open(filepath) as f:
            squad = json.load(f)
            for article in squad["data"]:
                title = article.get("title", "").strip()
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"].strip()
                    
                    if 'ans_ext' in tasks:
                        ans_ext_examples = self.process_ans_ext(paragraph)
                        for example in ans_ext_examples:
                                yield count, example
                                count += 1
                    
                    if 'e2e_qg' in tasks:
                        yield count, self.process_e2e_qg(paragraph)
                        count += 1
                    
                    for qa in paragraph["qas"]:
                        question = qa["question"].strip()
                        id_ = qa["id"]

                        answers = [answer["text"].strip() for answer in qa["answers"]]
                        for task in tasks:
                            if task == 'qa':
                                yield count, self.process_qa_text(context, question, answers[0])
                                count += 1
                            
                            if task == 'qg':
                                yield count, self.process_qg_text(context, question, qa["answers"][0])
                                count += 1

