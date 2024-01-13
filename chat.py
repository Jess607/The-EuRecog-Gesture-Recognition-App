from openai import OpenAI
from config import API_KEY

def llm_result(text):
    chatgpt_result = ''
    client=OpenAI(api_key=API_KEY)
    prompt = text
    model = "gpt-3.5-turbo-instruct"
    response = client.completions.create(model=model, prompt=prompt, max_tokens=70)
    generated_text =  response.choices[0].text
    chatgpt_result += generated_text+"\n"
    mytext = chatgpt_result
    
    return mytext




