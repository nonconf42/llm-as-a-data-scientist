from openai import OpenAI
import os

class LLM:
    def __init__(self, llm_name):
        self.llm_name = llm_name
        if llm_name == "gpt":
            api_key = os.environ.get('OPENAI_API_KEY')
            self.llm_model = OpenAI(api_key=api_key)

    def llm_call(self, prompt, model_name="gpt-4o"):
        if self.llm_name == "gpt":
            output_text = self.llm_model.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            output_text = output_text.choices[0].message.content
        return output_text