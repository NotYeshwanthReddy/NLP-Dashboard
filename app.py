from flask import Flask, request
from flask_cors import CORS

import torch
from torch import softmax

from transformers import BertTokenizer, BertForMaskedLM, GPT2Tokenizer, GPT2LMHeadModel
import nltk

import numpy as np

import json


app = Flask(__name__)
CORS(app)
top = 5


""" BERT """
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Fill in the Blanks
bert_model_mlm = BertForMaskedLM.from_pretrained('bert-base-uncased', output_attentions=True)
bert_model_mlm.eval()
# Question Answering
bert_model_qna = BertForQuestionAnswering.from_pretrained('bert-base-uncased', output_attentions=True)
bert_model_qna.eval()

""" GPT2 """
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# Language Modelling
gpt_model_lm = GPT2LMHeadModel.from_pretrained('gpt2', output_attentions=True)
gpt_model_lm.eval()


@app.route('/get_next_word', methods=['POST'])
def LM_predict():
  sentence_orig = request.form.get('text')

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  encoded_prompt = gpt_tokenizer.encode(sentence_orig, add_special_tokens=False, return_tensors="pt")
  encoded_prompt = encoded_prompt.to(device)

  if encoded_prompt.size()[-1] == 0:
    input_ids = None
  else:
    input_ids = encoded_prompt


  output_sequences = gpt_model_lm.generate(
    input_ids=input_ids,
    max_length= 1 + len(encoded_prompt[0]),
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    repetition_penalty=1.0,
    do_sample=True,
    num_return_sequences=5,
  )

  output = []
  for generated_sequence in output_sequences:
    # print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
    generated_sequence = generated_sequence.tolist()
    # Decode text
    text = gpt_tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
    # Remove all text after the stop token
    text = text[: text.find(None) if None else None]
    # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
    total_sequence = (
      text[len(gpt_tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
    )
    output.append(total_sequence)
  return '<p id="info">'+ output +'</p>'


@app.route('/fillblanks', methods=['POST'])
def MLM_predict():
  sentence_orig = request.form.get('text')
  if '__' not in sentence_orig:
    return sentence_orig

  # formatting sentence
  sentence = sentence_orig.replace('__', '[MASK]')

  # tokenising sentence
  # tokenized_text = bert_tokenizer.tokenize(sentence)
  # encoding sentence
  input_ids = torch.tensor(bert_tokenizer.encode(sentence)).unsqueeze(0)
  # Getting the outputs
  outputs = bert_model_mlm(input_ids)[0]

  # Getting the index of the masked words in sentence
  masked_index = []
  for i,j in enumerate(input_ids[0]):
    if j == bert_tokenizer.convert_tokens_to_ids("[MASK]"):
      masked_index.append(i)

  # Getting suggested words and their probabilities for each masked word
  answers = []
  for index in masked_index:
    predicted_prob,predicted_index = torch.topk(torch.softmax(outputs[0][index],dim=0),top)
    predicted_tokens = bert_tokenizer.convert_ids_to_tokens(predicted_index)
    answers.append(list(zip(predicted_tokens,predicted_prob.tolist())))

  # Preparing the output for HTML
  for answer in answers:
    popup = [str(word)+": "+str(round(value, 4)) for word,value in answer]
    popup = ("<br>").join(popup)
    sentence_orig = sentence_orig.replace('__', '<i><b>'+answer[0][0]+'</b><span>'+popup+'</span></i>', 1)
  output = '<p id="info">'+sentence_orig+'</p>'

  return output


@app.route('/qna', methods=['POST'])
def QnA_predict():
  text = request.form.get('text')
  passage = text['passage']
  question = text['question']

  # encoding question and passage using tokenizer
  input_ids = bert_tokenizer.encode(question, passage)
  token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
  start_scores, end_scores = bert_model_qna(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
  all_tokens = bert_tokenizer.convert_ids_to_tokens(input_ids)
  # Calculating the confidence score
  confidence_score = ((torch.max(softmax(start_scores, dim = 1))+torch.max(softmax(end_scores, dim = 1)))/2)

  # conditions to be taken care manually 
  # (end is before start)
  try:
    if torch.argmax(start_scores) < all_tokens.index("[SEP]"):
      if all_tokens.index("[SEP]") < torch.argmax(end_scores)+1:
        answer = ' '.join(all_tokens[all_tokens.index("[SEP]") : torch.argmax(end_scores)+1]).replace(" ##","")
      else:
        answer="[SEP]"
    else:
      answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]).replace(" ##","")

  except:
    answer ="[SEP]"

  # Preparing the output for HTML
  answer = '<p id="info"><b><i>'+answer.replace(' ##', '') + '</b><span>'+str(confidence_score.data).split("(")[-1][:-1]+'</span></i></p>'
  return answer


if __name__=='__main__':
	app.run(debug=False)
