from flask import Flask, request
from flask_cors import CORS
import torch
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM
import nltk

app = Flask(__name__)
CORS(app)
top = 5

# BERT - Fill in the Blanks
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased', output_attentions=True)
bert_model.eval()

# GPT2 - Language Modelling
gpt_tokenizer = BertTokenizer.from_pretrained('')
gpt_model = BertForMaskedLM.from_pretrained('', output_attentions=True)
gpt_model.eval()

# @app.route('/complete_sentence', methods=['POST'])
# def LM_predict():
# 	sentence_orig = request.form.get('text')
# 	if '____' not in sentence_orig:
# 		return sentence_orig

# 	sentence = sentence_orig.replace('____', 'MASK')
# 	tokens = nltk.word_tokenize(sentence)
# 	sentences = nltk.sent_tokenize(sentence)
# 	sentence = " [SEP] ".join(sentences)
# 	sentence = "[CLS] " + sentence + " [SEP]"
# 	tokenized_text = gpt_tokenizer.tokenize(sentence)
# 	masked_index = tokenized_text.index('mask')
# 	tokenized_text[masked_index] = "[MASK]"
# 	indexed_tokens = gpt_tokenizer.convert_tokens_to_ids(tokenized_text)

# 	segments_ids = []
# 	sentences = sentence.split('[SEP]')
# 	for i in range(len(sentences)-1):
# 		segments_ids.extend([i]*len(sentences[i].strip().split()))
# 		segments_ids.extend([i])

# 	tokens_tensor = torch.tensor([indexed_tokens])
# 	segments_tensors = torch.tensor([segments_ids])

# 	with torch.no_grad():
# 	    outputs = gpt_model(tokens_tensor, token_type_ids=segments_tensors)
# 	    predictions = outputs[0]
# 	    attention = outputs[-1]

# 	dim = attention[2][0].shape[-1]*attention[2][0].shape[-1]
# 	a = attention[2][0].reshape(12, dim)
# 	b = a.mean(axis=0)
# 	c = b.reshape(attention[2][0].shape[-1],attention[2][0].shape[-1])
# 	avg_wgts = c[masked_index]
# 	#print (avg_wgts, tokenized_text)
# 	focus = [tokenized_text[i] for i in avg_wgts.argsort().tolist()[::-1] if tokenized_text[i] not in ['[SEP]', '[CLS]', '[MASK]']][:5]

# 	# for layer in range(12):
# 	# 	weights_layer = np.array(attention[0][0][layer][masked_index])
# 	# 	print (weights_layer, tokenized_text)
# 	# 	print (weights_layer.argsort()[-3:][::-1])
# 	# 	print ()
# 	predicted_index = torch.argmax(predictions[0, masked_index]).item()
# 	predicted_token = gpt_tokenizer.convert_ids_to_tokens([predicted_index])[0]
# 	for f in focus:
# 		sentence_orig = sentence_orig.replace(f, '<font color="blue">'+f+'</font>')
# 	return sentence_orig.replace('____', '<font color="red"><b><i>'+predicted_token+'</i></b></font>')


@app.route('/fillblanks', methods=['POST'])
def MLM_predict():
	sentence_orig = request.form.get('text')
	if '__' not in sentence_orig:
		return sentence_orig

	sentence = sentence_orig.replace('__', '[MASK]')

	tokenized_text = bert_tokenizer.tokenize(sentence)
	input_ids = torch.tensor(bert_tokenizer.encode(sentence)).unsqueeze(0)
	outputs = bert_model(input_ids)[0]

	masked_index = []
	for i,j in enumerate(input_ids[0]):
		if j == bert_tokenizer.convert_tokens_to_ids("[MASK]"):
			masked_index.append(i)

	answers = []
	for index in masked_index:
		predicted_prob,predicted_index = torch.topk(torch.softmax(outputs[0][index],dim=0),top)
		predicted_tokens = bert_tokenizer.convert_ids_to_tokens(predicted_index)
		answers.append(list(zip(predicted_tokens,predicted_prob.tolist())))
	
	for answer in answers:
		popup = [str(word)+": "+str(round(value, 4)) for word,value in answer]
		popup = ("<br>").join(popup)
		sentence_orig = sentence_orig.replace('__', '<i><b>'+answer[0][0]+'</b><span>'+popup+'</span></i>', 1)
	output = '<p id="info">'+sentence_orig+'</p>'
	
	return output


if __name__=='__main__':
	app.run(debug=False)
