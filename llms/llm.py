from openai import OpenAI
import os
import openai

TOKEN_TRACKING_FILE = 'total_tokens.txt'

class LLM:
    def __init__(self, platform):
        self.platform = platform
        if platform == "openai":
            api_key = os.environ.get('OPENAI_API_KEY')
            self.llm_model = OpenAI(api_key=api_key)
        elif platform == 'deepinfra':
            api_key = os.environ.get('DEEPINFRA_API_KEY')
            self.llm_model = OpenAI(
                base_url="https://api.deepinfra.com/v1/openai",
                api_key=api_key
            )
     

    def llm_call(self, prompt, model_name="gpt-4o"):
        if self.platform == "openai":
            output_text = self.llm_model.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
                
            output_text = output_text.choices[0].message.content
        
        elif self.platform == 'deepinfra':
            output_text = self.llm_model.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            output_text = output_text.choices[0].message.content


        
        return output_text





