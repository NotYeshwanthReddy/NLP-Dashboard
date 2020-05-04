# NLP Dashboard

This is to showcase the Cutting edge NLP Models and their capabilities as-per current times (year 2019).

## Steps to Run
````
$> python3 app.py
````
* For LM task -
    * Open index.html in the browser and start typing :speech_balloon:
    * Suggestions are shown below
* For MLM task -
    * Find the MLM text box and enter your sentence with __ where required
    * __ represents a masked word which the model predicts.



## UI
![NLP Dashboard UI](https://github.com/NotYeshwanthReddy/NLP-Dashboard/blob/master/UI.png)

## Details
* __Masked Laguage Modelling Task:__
    * __Model__ - bert-base-uncased
    * __Pre-trained Task__ - MaskedLM
* __Laguage Modelling Task:__
    * __Model__ - gpt2
    * __Pre-trained Task__ - LM

## Technologies Used
1. PyTorch
2. HTML/Bootstrap
3. Flask
