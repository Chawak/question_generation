import itertools
import logging
from typing import Optional, Dict, Union
from functools import reduce
from nltk import sent_tokenize as en_sent_tokenize
from pythainlp.tokenize import word_tokenize,sent_tokenize
from pythainlp.util import countthai
import tensorflow as tf
import torch
from transformers import(
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    set_seed,
    pipeline as ner_pipeline
)
import numpy as np
from flair.data import Sentence
from flair.models import SequenceTagger
from ner_utils import wangchan_ner_result_to_extracted_answer, preprocess_for_wangchan
from ranking import get_ranking_score
from pipelines_util import (
    get_best_match_qa,
    AD_BE_convert,
    add_unit_to_answer,
    create_custom_tokenizer,
    spell_correct)
model_name = "wangchanberta-base-att-spm-uncased" 

#create tokenizer
dataset_name = "lst20"
tf.random.set_seed(42)
set_seed(42)
logger = logging.getLogger(__name__)

class QGPipeline:
    """Poor man's QG pipeline"""
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        ans_model: PreTrainedModel,
        ans_tokenizer: PreTrainedTokenizer,
        qg_format: str,
        use_cuda: bool
    ):
        
        self.th_ner_tokenizer = AutoTokenizer.from_pretrained(
                f'airesearch/{model_name}' ,
                revision='main',
                model_max_length=416,)

        self.th_ner_pipeline =  ner_pipeline(task='ner',
            tokenizer=self.th_ner_tokenizer,
            model = f'airesearch/{model_name}' ,
            revision = f'finetuned@{dataset_name}-ner',
            ignore_labels=[], 
            grouped_entities=True)
        
        self.en_ner_pipeline = SequenceTagger.load("flair/ner-english-ontonotes-large")

        self.corrector_tokenizer ,self.corrector_dict = create_custom_tokenizer()

        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer

        self.ans_model = ans_model
        self.ans_model.eval()
        self.ans_tokenizer = ans_tokenizer

        self.qg_format = qg_format

        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.to(self.device)

        if self.ans_model is not self.model:
            self.ans_model.to(self.device)

        assert self.model.__class__.__name__ in ["T5ForConditionalGeneration", "BartForConditionalGeneration","MT5ForConditionalGeneration","MBartForConditionalGeneration"]
        
        if "T5ForConditionalGeneration" in self.model.__class__.__name__:
            self.model_type = "t5"
        elif "MT5ForConditionalGeneration" in self.model.__class__.__name__:
            self.model_type = "mt5"
        elif "MBartForConditionalGeneration" in self.model.__class__.__name__:
            self.model_type = "mbart"
        else:
            self.model_type = "bart"

    def __call__(self, inputs: str,generate_mode="diverse_beam_search",num_question=3,generate_batch_size=5):
        inputs = " ".join(inputs.split())
        
        is_th=False        
        if countthai(inputs)>0:
            is_th=True
        
        sents, answers = self._extract_answers(inputs,is_th)
        flat_answers = list(itertools.chain(*answers))
        if len(flat_answers) == 0:
          return []

        sents, answers = self._extract_answers(inputs,is_th)
        _,ner_answers =self._extract_answers_ner(inputs,is_th)
        
        answers = [self.filter_answer(inputs,answers[i]+ner_answers[i]) for i in range(len(answers))]

        if self.qg_format == "prepend":
            qg_examples = self._prepare_inputs_for_qg_from_answers_prepend(inputs, answers)
        else:
            qg_examples = self._prepare_inputs_for_qg_from_answers_hl(sents, answers)
        qg_inputs = [example['source_text'] for example in qg_examples]
        questions=[]
        bs=generate_batch_size
        for i in range(0,len(qg_inputs)//bs + (1 if len(qg_inputs)%bs else 0)):
            tmp_questions = self._generate_questions(qg_inputs[i*bs:(i+1)*bs],generate_mode,is_th,num_question)
            questions+=tmp_questions
        
        output = [{'answer': example['answer'], 'question': [q.strip() for q in que[0]],"prob_score":[q for q in que[1]]} for example, que in zip(qg_examples, questions)]

        return output
    
    def filter_answer(self,context,ans_list):
        return [ans for ans in ans_list if ans in context]

    def generate_question_from_context_and_answer_prepend(self, context,answers,generate_mode="diverse_beam_search",num_question=3,generate_batch_size=5):
        
        is_th=False        
        if countthai(context)>0:
            is_th=True

        qg_examples = self._prepare_inputs_for_qg_from_answers_prepend(context, answers)
        qg_inputs = [example['source_text'] for example in qg_examples]
        questions=[]
        bs=generate_batch_size
        if generate_mode=="sample" :
            bs=1
        for i in range(0,len(qg_inputs)//bs + (1 if len(qg_inputs)%bs else 0)):
            
            tmp_questions = self._generate_questions(qg_inputs[i*bs:(i+1)*bs],generate_mode,is_th,num_question)
            questions+=tmp_questions
        output = [{'answer': example['answer'], 'question': [q.strip() for q in que[0]],"prob_score":[q for q in que[1]]} for example, que in zip(qg_examples, questions)]

        return output

    def deduplicate_question(self,context,answer,question_list,prob_list,num_question,is_th):
        q_set = set()
        new_q = []
        new_score = []
        for i,que in enumerate(question_list):
          tmp_que = que.replace(" ","")
          if tmp_que=="":
            continue
          if tmp_que[-1]!="?":
            tmp_que+="?"
          if tmp_que not in q_set:
            q_set.add(tmp_que)
            new_q.append(que)
            new_score.append(prob_list[i])
        
        scored_question=get_ranking_score(context,new_q,answer,is_th)
        
        if is_th :
            ans=[q for s,q in sorted(scored_question)[::-1][:num_question]]
            return ans,[np.e**(new_score[new_q.index(q)]) for q in ans]
        
        return new_q[:num_question],[np.e**(s) for s in new_score[:num_question]]

    def flatten(self,nested_list):
  
        nested_list = reduce(lambda x,y: x+y, nested_list)
        return nested_list

    def _generate_questions(self, inputs,generate_mode,is_th,num_question=3):
        inputs_answers=[]
        for inp in inputs:
          answer_start=inp.find("answer: ")+8
          answer_end=inp.find("context: ")
          inputs_answers.append(inp[answer_start:answer_end])
        inputs_context=inputs[0][inputs[0].find("context: ")+9:]
        inputs = self._tokenize(inputs, padding=True, truncation=True)

        num_over_generated = num_question*3
        prob_score=[[0]*num_over_generated]
        if generate_mode =="augmented":
            outs = self.model.generate(
                input_ids=inputs['input_ids'].to(self.device), 
                attention_mask=inputs['attention_mask'].to(self.device), 
                max_length=64,
                num_beams=max(12,num_over_generated),
                repetition_penalty=2.0,
                no_repeat_ngram_size=7
            )
            prob_score=[[0]]*num_over_generated
            questions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
            questions = [que.split("<sep>") for que in questions]

        elif generate_mode=="top_n_sequence":
            outs = self.model.generate(
                input_ids=inputs['input_ids'].to(self.device), 
                attention_mask=inputs['attention_mask'].to(self.device), 
                max_length=96,
                num_beams=12,
                num_return_sequences=num_over_generated,
                output_scores=True,
                return_dict_in_generate=True
            )
            
            prob_score=outs.sequences_scores.tolist()
            outs=outs.sequences
            questions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
            
            questions = [questions[i:i+num_over_generated] for i in range(0,len(questions),num_over_generated)]
            prob_score = [prob_score[i:i+num_over_generated] for i in range(0,len(prob_score),num_over_generated)]

        elif generate_mode=="diverse_beam_search":
            if num_over_generated%2==1:
                num_over_generated+=1
            outs = self.model.generate(
                input_ids=inputs['input_ids'].to(self.device), 
                attention_mask=inputs['attention_mask'].to(self.device), 
                max_length=64,
                # num_beams=max(12,num_over_generated if num_over_generated%2==0 else num_over_generated+1),
                num_beams=max(12,num_over_generated),
                num_beam_groups=2,
                diversity_penalty=3.0,
                num_return_sequences=num_over_generated,
                output_scores=True,
                return_dict_in_generate=True
            )
            prob_score=outs.sequences_scores.tolist()
            outs=outs.sequences
            questions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
            
            questions = [questions[i:i+num_over_generated] for i in range(0,len(questions),num_over_generated)]
            prob_score = [prob_score[i:i+num_over_generated] for i in range(0,len(prob_score),num_over_generated)]
            
        elif generate_mode=="sample": 
            tf.random.set_seed(42)
            set_seed(42)
            outs = self.model.generate(
                input_ids=inputs['input_ids'].to(self.device), 
                attention_mask=inputs['attention_mask'].to(self.device), 
                max_length=64,
                do_sample=True,
                top_k=5, 
                temperature=1.20,
                num_return_sequences=num_over_generated,
            )

            questions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
            questions = [questions[i:i+num_over_generated] for i in range(0,len(questions),num_over_generated)]
        else:
            outs = self.model.generate(
                input_ids=inputs['input_ids'].to(self.device), 
                attention_mask=inputs['attention_mask'].to(self.device), 
                max_length=64,
                num_beams=4,
                output_scores=True,
                return_dict_in_generate=True
            )
            prob_score=outs.sequences_scores.tolist()
            outs=outs.sequences
            questions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]

            questions = [que.split("<sep>") for que in questions]
            prob_score = [[p] for p in prob_score]

        questions=[self.deduplicate_question(inputs_context,inputs_answers[i],que,prob_score[i],num_question,is_th) for i,que in enumerate(questions)]
        
          
        return questions
    
    def _extract_answers(self, context,is_th):
        sents, inputs = self._prepare_inputs_for_ans_extraction(context,is_th)
        bs=32
        
        real_answers=[]
        
        for i in range(0,len(inputs)//bs + (1 if len(inputs)%bs else 0)):
            
            batch_inputs = self._tokenize(inputs[i*bs:(i+1)*bs], padding=True, truncation=True)
            outs = self.ans_model.generate(
                input_ids=batch_inputs['input_ids'].to(self.device), 
                attention_mask=batch_inputs['attention_mask'].to(self.device), 
                max_length=32,
            )
            dec = [self.ans_tokenizer.decode(ids, skip_special_tokens=False) for ids in outs]
            answers = [item.replace("<pad> ","").replace("<pad>","").split('<sep>') for item in dec]
            answers = [i[:-1] for i in answers]
            answers = [[t.strip() for t in i] for i in answers ]
            real_answers+=answers
        
        return sents, answers
    
    def get_th_ner(self,sentence):
        return wangchan_ner_result_to_extracted_answer(self.th_ner_pipeline(preprocess_for_wangchan(sentence)),sentence)

    def _extract_answers_ner(self, context,is_th):
        answers = []
        if is_th:
            sents = sent_tokenize(context)
            for sent in sents:
                answers.append(self.get_th_ner(sent))
        else :
            sents = en_sent_tokenize(context)
            for sent in sents:
                sentence = Sentence(sent)
                self.en_ner_pipeline.predict(sentence)
                answers.append([x.text for x in sentence.get_spans('ner')])
        return sents, answers

    def _tokenize(self,
        inputs,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=512
    ):
        inputs = self.tokenizer.batch_encode_plus(
            inputs, 
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt"
        )
        return inputs
    
    def _prepare_inputs_for_ans_extraction(self, text, is_th):
        if is_th:
            sents = sent_tokenize(text)
        else :
            sents = en_sent_tokenize(text)


        inputs = []
        for i in range(len(sents)):
            source_text = "extract answers:"
            for j, sent in enumerate(sents):
                if i == j:
                    sent = "<hl> %s <hl>" % sent
                source_text = "%s %s" % (source_text, sent)
                source_text = source_text.strip()
            
            if self.model_type == "t5" or self.model_type == "mt5":
                source_text = source_text + " </s>"
            inputs.append(source_text)

        return sents, inputs
    
    def _prepare_inputs_for_qg_from_answers_hl(self, sents, answers):
        inputs = []
        for i, answer in enumerate(answers):
            if len(answer) == 0: continue
            for answer_text in answer:
                sent = sents[i]
                sents_copy = sents[:]
                
                answer_text = answer_text.strip()
                ans_start_idx = sent.index(answer_text)
                
                sent = f"{sent[:ans_start_idx]} <hl> {answer_text} <hl> {sent[ans_start_idx + len(answer_text): ]}"
                sents_copy[i] = sent
                
                source_text = " ".join(sents_copy)
                source_text = f"generate question: {source_text}" 
                if self.model_type == "t5" or self.model_type == "mt5":
                    source_text = source_text + " </s>"
                
                inputs.append({"answer": answer_text, "source_text": source_text})
        
        return inputs
    
    def _prepare_inputs_for_qg_from_answers_prepend(self, context, answers):
        flat_answers = list(set(list(itertools.chain(*answers))))
        examples = []
        
        for answer in flat_answers:
            source_text = f"answer: {answer} context: {context}"
            if self.model_type == "t5" or self.model_type == "mt5":
                source_text = source_text + " </s>"
            
            examples.append({"answer": answer, "source_text": source_text})
        return examples

    
class MultiTaskQAQGPipeline(QGPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, inputs: Union[Dict, str],generate_mode="diverse_beam_search",num_question=3,generate_batch_size=5):
        if type(inputs) is str:
            # do qg
            return super().__call__(inputs,generate_mode,num_question,generate_batch_size)
        else:
            # do qa
            
            if inputs["task"]=="qa":
                
                if "use_threshold" not in inputs:
                    inputs["use_threshold"]=False
                if "threshold" not in inputs:
                    inputs["threshold"]=0.96
                if "use_text_search" not in inputs:
                    inputs["use_text_search"]=False
                if "correct_before_pred" not in inputs:
                    inputs["correct_before_pred"]=True
                return self.question_answering(inputs["question"], inputs["context"],inputs["use_text_search"],inputs["correct_before_pred"])
            else :
                return super().generate_question_from_context_and_answer_prepend(inputs["context"],inputs["answers"],generate_mode,num_question,generate_batch_size)
            
    
    def _prepare_inputs_for_qa(self, question, context):
        source_text = f"question: {question}  context: {context}"
        if self.model_type == "t5" or self.model_type == "mt5":
            source_text = source_text + " </s>"
        return  source_text
    
    def question_answering(self, question, context, use_text_search, correct_before_pred, use_threshold,threshold):

        if correct_before_pred:
            question=spell_correct(question,self.corrector_tokenizer,self.corrector_dict)
            context=spell_correct(context,self.corrector_tokenizer,self.corrector_dict)

        source_text = self._prepare_inputs_for_qa(question, context)
        inputs = self._tokenize([source_text], padding=False)
    
        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device), 
            attention_mask=inputs['attention_mask'].to(self.device), 
            max_length=16,
            num_beams=4,
            output_scores=True,
            return_dict_in_generate=True,
            num_return_sequences=3
        )
        
        prob_score=outs.sequences_scores.tolist()
        
        ans_prob_score=prob_score[0]
        outs=outs.sequences
        answer = self.tokenizer.decode(outs[0], skip_special_tokens=True)
        
        answer=answer.replace("ํา","ำ")

        if use_threshold:
            if answer=="ไม่มีคำตอบ" and prob_score<threshold:
                answer=self.tokenizer.decode(outs[1], skip_special_tokens=True)
                ans_prob_score=prob_score[1]

        
        if use_text_search:
            
            answer = add_unit_to_answer(context,AD_BE_convert(context,answer),question)
            if answer not in context and answer!="ไม่มีคำตอบ":
                answer = get_best_match_qa(answer,context,step=1,flex=len(answer)//2-1)[0]

        return answer,np.e**ans_prob_score


class E2EQGPipeline:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        use_cuda: bool
    ) :

        self.model = model
        self.tokenizer = tokenizer

        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.to(self.device)

        assert self.model.__class__.__name__ in ["T5ForConditionalGeneration", "BartForConditionalGeneration","MT5ForConditionalGeneration","MBartForConditionalGeneration"]
        
        if "T5ForConditionalGeneration" in self.model.__class__.__name__:
            self.model_type = "t5"
        elif "MT5ForConditionalGeneration" in self.model.__class__.__name__:
            self.model_type = "mt5"
        elif "MBartForConditionalGeneration" in self.model.__class__.__name__:
            self.model_type = "mbart"
        else:
            self.model_type = "bart"
        
        self.default_generate_kwargs = {
            "max_length": 256,
            "num_beams": 4,
            "length_penalty": 1.5,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }
    
    def __call__(self, context: str, **generate_kwargs):
        inputs = self._prepare_inputs_for_e2e_qg(context)

        # TODO: when overrding default_generate_kwargs all other arguments need to be passsed
        # find a better way to do this
        if not generate_kwargs:
            generate_kwargs = self.default_generate_kwargs
        
        input_length = inputs["input_ids"].shape[-1]
        
        # max_length = generate_kwargs.get("max_length", 256)
        # if input_length < max_length:
        #     logger.warning(
        #         "Your max_length is set to {}, but you input_length is only {}. You might consider decreasing max_length manually, e.g. summarizer('...', max_length=50)".format(
        #             max_length, input_length
        #         )
        #     )

        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device), 
            attention_mask=inputs['attention_mask'].to(self.device),
            **generate_kwargs
        )

        prediction = self.tokenizer.decode(outs[0], skip_special_tokens=True)
        questions = prediction.split("<sep>")
        if len(questions)>1:
            questions=questions[:-1]
        questions = [question.strip() for question in questions]
        return questions
    
    def _prepare_inputs_for_e2e_qg(self, context):
        source_text = f"generate questions: {context}"
        if self.model_type == "t5" or self.model_type == "mt5":
            source_text = source_text + " </s>"
        
        inputs = self._tokenize([source_text], padding=False)
        return inputs
    
    def _tokenize(
        self,
        inputs,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=512
    ):
        inputs = self.tokenizer.batch_encode_plus(
            inputs, 
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt"
        )
        return inputs


SUPPORTED_TASKS = {
    "question-generation": {
        "impl": QGPipeline,
        "default": {
            "model": "valhalla/t5-small-qg-hl",
            "ans_model": "valhalla/t5-small-qa-qg-hl",
        }
    },
    "multitask-qa-qg": {
        "impl": MultiTaskQAQGPipeline,
        "default": {
            "model": "valhalla/t5-small-qa-qg-hl",
        }
    },
    "e2e-qg": {
        "impl": E2EQGPipeline,
        "default": {
            "model": "valhalla/t5-small-e2e-qg",
        }
    }
}

def pipeline(
    task: str,
    model: Optional = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    qg_format: Optional[str] = "highlight",
    ans_model: Optional = None,
    ans_tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    use_cuda: Optional[bool] = True,
    **kwargs,
):
    # Retrieve the task
    if task not in SUPPORTED_TASKS:
        raise KeyError("Unknown task {}, available tasks are {}".format(task, list(SUPPORTED_TASKS.keys())))

    targeted_task = SUPPORTED_TASKS[task]
    task_class = targeted_task["impl"]

    # Use default model/config/tokenizer for the task if no model is provided
    if model is None:
        model = targeted_task["default"]["model"]
    
    # Try to infer tokenizer from model or config name (if provided as str)
    if tokenizer is None:
        if isinstance(model, str):
            tokenizer = model
        else:
            # Impossible to guest what is the right tokenizer here
            raise Exception(
                "Impossible to guess which tokenizer to use. "
                "Please provided a PretrainedTokenizer class or a path/identifier to a pretrained tokenizer."
            )
    
    # Instantiate tokenizer if needed
    if isinstance(tokenizer, (str, tuple)):
        if isinstance(tokenizer, tuple):
            # For tuple we have (tokenizer name, {kwargs})
            tokenizer = AutoTokenizer.from_pretrained(tokenizer[0], **tokenizer[1])
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    
    # Instantiate model if needed
    if isinstance(model, str):
        model = AutoModelForSeq2SeqLM.from_pretrained(model)
    
    if task == "question-generation":
        if ans_model is None:
            # load default ans model
            ans_model = targeted_task["default"]["ans_model"]
            ans_tokenizer = AutoTokenizer.from_pretrained(ans_model)
            ans_model = AutoModelForSeq2SeqLM.from_pretrained(ans_model)
        else:
            # Try to infer tokenizer from model or config name (if provided as str)
            if ans_tokenizer is None:
                if isinstance(ans_model, str):
                    ans_tokenizer = ans_model
                else:
                    # Impossible to guest what is the right tokenizer here
                    raise Exception(
                        "Impossible to guess which tokenizer to use. "
                        "Please provided a PretrainedTokenizer class or a path/identifier to a pretrained tokenizer."
                    )
            
            # Instantiate tokenizer if needed
            if isinstance(ans_tokenizer, (str, tuple)):
                if isinstance(ans_tokenizer, tuple):
                    # For tuple we have (tokenizer name, {kwargs})
                    ans_tokenizer = AutoTokenizer.from_pretrained(ans_tokenizer[0], **ans_tokenizer[1])
                else:
                    ans_tokenizer = AutoTokenizer.from_pretrained(ans_tokenizer)

            if isinstance(ans_model, str):
                ans_model = AutoModelForSeq2SeqLM.from_pretrained(ans_model)
    
    if task == "e2e-qg":
        return task_class(model=model, tokenizer=tokenizer, use_cuda=use_cuda)
    elif task == "question-generation":
        return task_class(model=model, tokenizer=tokenizer, ans_model=ans_model, ans_tokenizer=ans_tokenizer, qg_format=qg_format, use_cuda=use_cuda)
    else:
        return task_class(model=model, tokenizer=tokenizer, ans_model=model, ans_tokenizer=tokenizer, qg_format=qg_format, use_cuda=use_cuda)
