#MODEL
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import transformers
from transformers import pipeline
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate

TOKEN = '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'

#PROMPT_TEMPLATE = """Com base no contexto e nas informações fornecidas pelo usuário, recomende produtos que atendam às suas necessidades, incluindo nome e especificações do item. Se você não souber a resposta, apenas diga que não sabe, não tente inventar uma resposta.  
  
#Contexto: {context}  
#Informações do usuário: {user_input}  
  
#Recomendações de produtos detalhadas:"""  
#PROMPT = PromptTemplate(  
#    template=PROMPT_TEMPLATE, input_variables=["context", "user_input"]  
#)  
#chain_type_kwargs = {"prompt": PROMPT}

class TokenizerModel():
    def __init__(self, model_name='meta-llama/Llama-2-7b-chat-hf'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       use_auth_token=TOKEN)
class TextModel():
    def __init__(self, model_name='meta-llama/Llama-2-7b-chat-hf'):
        self.textModel = AutoModelForCausalLM.from_pretrained(model_name,
                                                              torch_dtype=torch.float32,
                                                              use_auth_token=TOKEN)
        
class Pipeline():
    def __init__(self, task='text-generation',
                 precision=torch.bfloat16,
                 device='auto',
                 max_tokens=512,
                 min_tokens=-1,
                 temperature=0.0001,
                 top_p=0.95,
                 repetition_penalty=1.15, 
                 vector_db=None):
        self.task = task
        self.precision = precision
        self.device = device
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.model = TextModel().textModel
        self.tokenizer = TokenizerModel().tokenizer
        self.pipe = pipeline(task=self.task,
                             model=self.model,
                             tokenizer=self.tokenizer,
                             #torch_dtype=self.precision,
                             max_new_tokens=self.max_tokens,
                             min_new_tokens=self.min_tokens,
                             temperature=self.temperature,
                             top_p=self.top_p,
                             repetition_penalty=self.repetition_penalty)
        self.llm = HuggingFacePipeline(pipeline=self.pipe, model_kwargs={'temperature':0.0001})
        self.vector_db = vector_db
        self.retriever = self.vector_db.as_retriever(search_kwargs={'k':1})
        self.search = RetrievalQA.from_chain_type(llm=self.llm,
                                  chain_type="stuff",
                                  retriever=self.retriever,
                                  #achain_type_kwargs=chain_type_kwargs,
                                  return_source_documents=True)