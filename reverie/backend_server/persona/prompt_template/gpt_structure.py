"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling Local LLM APIs (替换OpenAI).
"""
import json
import random
import time
import sys
import os

# Add path to import our local AI service
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'ai_service'))

from utils import *

try:
    from ai_service import get_ai_service
    LOCAL_LLM_AVAILABLE = True
    print("[gpt_structure] Successfully imported local AI service")
except ImportError as e:
    LOCAL_LLM_AVAILABLE = False
    print(f"[gpt_structure] Failed to import local AI service: {e}")

# Initialize global AI service
if LOCAL_LLM_AVAILABLE:
    _ai_service = get_ai_service(enable_optimizations=True)
else:
    _ai_service = None

def temp_sleep(seconds=0.1):
  time.sleep(seconds)

def ChatGPT_single_request(prompt): 
  temp_sleep()
  
  if not LOCAL_LLM_AVAILABLE or _ai_service is None:
    return "Local LLM service not available"
  
  try:
    response = _ai_service.generate(prompt, use_optimizations=True)
    return response
  except Exception as e:
    print(f"[ChatGPT_single_request] Error: {e}")
    return "Local LLM generation error"


# ============================================================================
# #####################[SECTION 1: CHATGPT-3 STRUCTURE] ######################
# ============================================================================

def GPT4_request(prompt): 
  """
  Given a prompt, make a request to Local LLM server and returns the response. 
  ARGS:
    prompt: a str prompt
  RETURNS: 
    a str of Local LLM's response. 
  """
  temp_sleep()

  if not LOCAL_LLM_AVAILABLE or _ai_service is None:
    print("Local LLM ERROR - Service not available")
    return "Local LLM ERROR"

  try: 
    response = _ai_service.generate(prompt, use_optimizations=True)
    return response
  except Exception as e: 
    print(f"Local LLM ERROR: {e}")
    return "Local LLM ERROR"


def ChatGPT_request(prompt): 
  """
  Given a prompt, make a request to Local LLM server and returns the response. 
  ARGS:
    prompt: a str prompt
  RETURNS: 
    a str of Local LLM's response. 
  """
  # temp_sleep()
  
  if not LOCAL_LLM_AVAILABLE or _ai_service is None:
    print("Local LLM ERROR - Service not available")
    return "Local LLM ERROR"
  
  try: 
    response = _ai_service.generate(prompt, use_optimizations=True)
    return response
  except Exception as e: 
    print(f"Local LLM ERROR: {e}")
    return "Local LLM ERROR"


def GPT4_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_gpt_response = GPT4_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass

  return False


def ChatGPT_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  # prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]

      # print ("---ashdfaf")
      # print (curr_gpt_response)
      # print ("000asdfhia")
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass

  return False


def ChatGPT_safe_generate_response_OLD(prompt, 
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 
    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      if verbose: 
        print (f"---- repeat count: {i}")
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass
  print ("FAIL SAFE TRIGGERED") 
  return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

def GPT_request(prompt, gpt_parameter): 
  """
  Given a prompt and parameters, make a request to Local LLM server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with parameters (for compatibility)   
  RETURNS: 
    a str of Local LLM's response. 
  """
  temp_sleep()
  
  if not LOCAL_LLM_AVAILABLE or _ai_service is None:
    print("TOKEN LIMIT EXCEEDED - Local LLM not available")
    return "TOKEN LIMIT EXCEEDED"
  
  try: 
    # Use local LLM with optimizations
    response = _ai_service.generate(prompt, use_optimizations=True)
    return response
  except Exception as e: 
    print(f"TOKEN LIMIT EXCEEDED - Local LLM error: {e}")
    return "TOKEN LIMIT EXCEEDED"


def generate_prompt(curr_input, prompt_lib_file): 
  """
  Takes in the current input (e.g. comment that you want to classifiy) and 
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this 
  function replaces this substr with the actual curr_input to produce the 
  final promopt that will be sent to the GPT3 server. 
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file. 
  RETURNS: 
    a str prompt that will be sent to OpenAI's GPT server.  
  """
  if type(curr_input) == type("string"): 
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):   
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt: 
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


def safe_generate_response(prompt, 
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False): 
  if verbose: 
    print (prompt)

  for i in range(repeat): 
    curr_gpt_response = GPT_request(prompt, gpt_parameter)
    if func_validate(curr_gpt_response, prompt=prompt): 
      return func_clean_up(curr_gpt_response, prompt=prompt)
    if verbose: 
      print ("---- repeat count: ", i, curr_gpt_response)
      print (curr_gpt_response)
      print ("~~~~")
  return fail_safe_response


def get_embedding(text, model="local-embedding"):
  """
  Generate embedding using local embedding service
  """
  text = text.replace("\n", " ")
  if not text: 
    text = "this is blank"
  
  if not LOCAL_LLM_AVAILABLE or _ai_service is None:
    # Return a dummy embedding vector for compatibility
    print("[get_embedding] Local embedding service not available, returning dummy vector")
    return [0.0] * 1536  # Standard embedding dimension
  
  try:
    # Check if our AI service has embedding capability
    if hasattr(_ai_service, 'get_embedding'):
      return _ai_service.get_embedding(text)
    else:
      # Import embedding service directly
      try:
        from embedding_service import get_embedding_service
        embedding_service = get_embedding_service()
        return embedding_service.encode(text).tolist()
      except ImportError:
        print("[get_embedding] Embedding service not available, returning dummy vector")
        return [0.0] * 1536
  except Exception as e:
    print(f"[get_embedding] Error generating embedding: {e}")
    return [0.0] * 1536


if __name__ == '__main__':
  gpt_parameter = {"engine": "text-davinci-003", "max_tokens": 50, 
                   "temperature": 0, "top_p": 1, "stream": False,
                   "frequency_penalty": 0, "presence_penalty": 0, 
                   "stop": ['"']}
  curr_input = ["driving to a friend's house"]
  prompt_lib_file = "prompt_template/test_prompt_July5.txt"
  prompt = generate_prompt(curr_input, prompt_lib_file)

  def __func_validate(gpt_response): 
    if len(gpt_response.strip()) <= 1:
      return False
    if len(gpt_response.strip().split(" ")) > 1: 
      return False
    return True
  def __func_clean_up(gpt_response):
    cleaned_response = gpt_response.strip()
    return cleaned_response

  output = safe_generate_response(prompt, 
                                 gpt_parameter,
                                 5,
                                 "rest",
                                 __func_validate,
                                 __func_clean_up,
                                 True)

  print (output)




















