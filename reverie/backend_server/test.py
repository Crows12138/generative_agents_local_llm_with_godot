"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: test.py
Description: Test file for Local LLM APIs (替换OpenAI).
"""
import json
import random
import time 
import sys
import os

# Add path to import our local AI service
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ai_service'))

from utils import *

try:
    from ai_service import get_ai_service
    LOCAL_LLM_AVAILABLE = True
    print("[test.py] Successfully imported local AI service")
    _ai_service = get_ai_service(enable_optimizations=True)
except ImportError as e:
    LOCAL_LLM_AVAILABLE = False
    print(f"[test.py] Failed to import local AI service: {e}")
    _ai_service = None

def ChatGPT_request(prompt): 
  """
  Given a prompt, make a request to Local LLM server and returns the response. 
  ARGS:
    prompt: a str prompt
  RETURNS: 
    a str of Local LLM's response. 
  """
  
  if not LOCAL_LLM_AVAILABLE or _ai_service is None:
    print("Local LLM ERROR - Service not available")
    return "Local LLM ERROR"
  
  try: 
    response = _ai_service.generate(prompt, use_optimizations=True)
    return response
  except Exception as e: 
    print(f"Local LLM ERROR: {e}")
    return "Local LLM ERROR"

prompt = """
---
Character 1: Maria Lopez is working on her physics degree and streaming games on Twitch to make some extra money. She visits Hobbs Cafe for studying and eating just about everyday.
Character 2: Klaus Mueller is writing a research paper on the effects of gentrification in low-income communities.

Past Context: 
138 minutes ago, Maria Lopez and Klaus Mueller were already conversing about conversing about Maria's research paper mentioned by Klaus This context takes place after that conversation.

Current Context: Maria Lopez was attending her Physics class (preparing for the next lecture) when Maria Lopez saw Klaus Mueller in the middle of working on his research paper at the library (writing the introduction).
Maria Lopez is thinking of initating a conversation with Klaus Mueller.
Current Location: library in Oak Hill College

(This is what is in Maria Lopez's head: Maria Lopez should remember to follow up with Klaus Mueller about his thoughts on her research paper. Beyond this, Maria Lopez doesn't necessarily know anything more about Klaus Mueller) 

(This is what is in Klaus Mueller's head: Klaus Mueller should remember to ask Maria Lopez about her research paper, as she found it interesting that he mentioned it. Beyond this, Klaus Mueller doesn't necessarily know anything more about Maria Lopez) 

Here is their conversation. 

Maria Lopez: "
---
Output the response to the prompt above in json. The output should be a list of list where the inner lists are in the form of ["<Name>", "<Utterance>"]. Output multiple utterances in ther conversation until the conversation comes to a natural conclusion.
Example output json:
{"output": "[["Jane Doe", "Hi!"], ["John Doe", "Hello there!"] ... ]"}
"""

print (ChatGPT_request(prompt))












