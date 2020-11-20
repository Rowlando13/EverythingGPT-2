# this is an example for cortex release 0.19 and may not deploy correctly on other releases of cortex
import torch
import boto3
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import nltk
import os
import json

def s3_download(cortex_base, modelpath, s3bucket, s3):
    #making path in cortex directory
    os.makedirs(os.path.join(cortex_base, modelpath), exist_ok=True)
    #getting response object for s3 folder
    response = s3.list_objects_v2(Bucket=s3bucket, Prefix=modelpath)

    #downloading objects from s3 folder using response object as guide
    for s3_obj in response["Contents"]:
        s3_obj_key = s3_obj["Key"]
        s3.download_file(s3bucket, s3_obj_key, os.path.join(cortex_base, s3_obj_key))
        print("downloaded s3://{}/{} to {}/{}. \n".format(s3bucket, s3_obj_key, cortex_base, s3_obj_key), end = '')

def text_cleanup(text_gen, list_gen, mode):
    """Take text generated, Text generated as list of sentences and mode, returns string of finalized sentence """
    #generates only 1 sentence
    if len(list_gen)<=1:
        clean_text_gen=text_gen
    else:
        #getting truncating the generating text for each mode
        # first if else for case where completion[0] == punctuation, crude fix
        if text_gen[0] not in [".", "?", "!"]:
            if mode=='s-completion':
                clean_text_gen=list_gen[0]
            #not perfect, if give complete sentence, then gives them two sentences
            elif mode=="s-completion+":
                clean_text_gen=list_gen[0]+list_gen[1]
            else:
                line_break=text_gen.find("\n")
                if line_break !=-1:
                    clean_text_gen=text_gen[0:line_break-1]
                else:
                    clean_text_gen=text_gen
        else:
            if mode=='s-completion':
                clean_text_gen=list_gen[0]+list_gen[1]
            #not perfect, if give complete sentence, then gives them two sentences
            elif mode=="s-completion+":
                #if not enough sentences
                if len(list_gen)<=2:
                    clean_text_gen=list_gen[0]+list_gen[1]
                else:
                    clean_text_gen=list_gen[0]+list_gen[1]+list_gen[2]
            else:
                line_break=text_gen.find("\n")
                if line_break !=-1:
                    clean_text_gen=text_gen[0:line_break-1]
                else:
                    clean_text_gen=text_gen

    return clean_text_gen

class PythonPredictor:
    def __init__(self, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"using device: {self.device}")
        #credentials
        aws_access_key_id="null"
        aws_secret_access_key="null"
        cortex_base1="tmp1"
        modelpath1='gpt2_124m_base'
        cortex_base2="tmp2"
        modelpath2='gpt2_124m_base'
        s3bucket='deranged-parrot-models'

        #creating client and getting list of objects
        s3 = boto3.client("s3", aws_access_key_id=aws_access_key_id, \
                  aws_secret_access_key=aws_secret_access_key)
        #downloading model1
        s3_download(cortex_base=cortex_base1, modelpath=modelpath1, \
        s3bucket=s3bucket, s3=s3)
        #testing
        #s3_download(cortex_base=cortex_base2, modelpath=modelpath2, \
        #s3bucket=s3bucket, s3=s3)

        self.tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(cortex_base1, modelpath1)).to(self.device)
        #use_cache speeds up decoding by storing and repassing hidden states. Possible out of memory issue.
        self.model1 = GPT2LMHeadModel.from_pretrained(\
        os.path.join(cortex_base1, modelpath1), use_cache=True).to(self.device)
        #testing
        #self.model2= GPT2LMHeadModel.from_pretrained(\
        #os.path.join(cortex_base2, modelpath2), use_cache=True).to(self.device)
        #nltk sentence tokenizer is not in basic module
        nltk.download("punkt")

    def predict(self, payload):
        #Need to add code so python formatted text converts to multiline string
        #for data logging
        print(payload)

        #prevent accidently accepting True or False
        api_key=str(payload["api_key"])

        #verifying key
        if api_key != "dW8tB$j3yx&KvEvsP8QSt24&M2%QwYXD":
            final_answer={"text": "Invalid api_key", "truncated": "Invalid api_key"}

        else :
            #setting inputs for length
            if payload['mode']=='s-completion':
                gen_len=45
            elif payload['mode']=="s-completion+":
                gen_len=90
            else:
                gen_len=450

            #storing text for readibility and effeciency
            truncated=False
            raw_text=payload["text"]

            #tokenizing
            tokens = self.tokenizer.encode(raw_text, return_tensors="pt").to(self.device)
            #generating input length
            input_length = len(tokens[0])

            #if input is too long, then shortening so program will still run
            if input_length+gen_len > 1023:
                text = self.tokeizer.decode(tokens[-573:])
                tokens = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
                input_length = 573
                truncated=True
            #do nothing to text, tokens
            else :
                text = raw_text

            if payload["pred_name"]=="model1":
                prediction = self.model1.generate(tokens, min_length=input_length+gen_len-5, max_length=input_length+gen_len, temperature=payload["temperature"], repetition_penalty=payload["repetition_penalty"], top_k=payload["top_k"], top_p=payload["top_p"], do_sample=True, num_return_sequences=payload["batch_size"])
            else :
                prediction = self.model2.generate(tokens, min_length=input_length+gen_len-5, max_length=input_length+gen_len, temperature=payload["temperature"], repetition_penalty=payload["repetition_penalty"], top_k=payload["top_k"], top_p=payload["top_p"], do_sample=True, num_return_sequences=payload["batch_size"])

            #creating final answer
            final_answer={"text":{}, "truncated":truncated}

            #looping through batch
            for z in range(0, payload["batch_size"]):

                #getting just the generated text
                text_gen=self.tokenizer.decode(prediction[z][input_length:], skip_special_tokens=True)

                #parsing the generated text into sentences
                list_gen=nltk.sent_tokenize(text_gen)

                #cleaning up text
                final_answer["text"][z] = text_cleanup(text_gen, list_gen, payload["mode"])

        return final_answer
