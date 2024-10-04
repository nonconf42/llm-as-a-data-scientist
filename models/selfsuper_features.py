import pandas as pd
import numpy as np
import openai
import json
import ast
import traceback
from typing import Dict, List, Any

# Set your OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'

class GPTUnsupervisedAnalyzer:
    def __init__(self, df: pd.DataFrame, column_descriptions: Dict[str, str], max_iterations: int = 5):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The 'df' parameter must be a pandas DataFrame.")
        if not isinstance(column_descriptions, dict):
            raise ValueError("The 'column_descriptions' parameter must be a dictionary.")
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            raise ValueError("The 'max_iterations' parameter must be a positive integer.")
        
        self.df = df.copy()
        self.column_descriptions = column_descriptions
        self.max_iterations = max_iterations
        self.previous_hypotheses = []  # List of dicts with 'hypothesis' and 'worked' keys

    def generate_statistics(self) -> str:
        """
        Generates statistical summary of the DataFrame.
        """
        try:
            stats = self.df.describe(include='all').to_dict()
            stats_json = json.dumps(stats, default=str)
            return stats_json
        except Exception as e:
            print(f"Error generating statistics: {e}")
            return "{}"

    def generate_hypotheses(self, stats_json: str) -> List[str]:
        """
        Uses GPT to generate hypotheses based on column descriptions and statistics.
        """
        previous_hypotheses_text = "\n".join([
            f"- Hypothesis: {h['hypothesis']}\n  Worked: {h['worked']}"
            for h in self.previous_hypotheses
        ]) if self.previous_hypotheses else "None"

        prompt = f"""
You are a data scientist analyzing a dataset. Given the column descriptions, statistical summaries, and previous hypotheses with their outcomes, generate up to 3 new and improved hypotheses for potential patterns or insights in the data. The new hypotheses should take into account what was learned from previous attempts.

Types of hypotheses may include:

- Interesting clusters in the data.
- If-then rules (e.g., if X happens, then Y is likely to happen).
- Cause-effect relationships (e.g., A affects B in a certain way).
- Reasoning about latent variables that might explain patterns in the data.
- Specific situations that might be interesting to explore.

Column Descriptions:
{json.dumps(self.column_descriptions, indent=2)}

Statistical Summaries:
{stats_json}

Previous Hypotheses and Outcomes:
{previous_hypotheses_text}

Provide the new hypotheses as a numbered list.
"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                n=1,
                stop=None,
                temperature=0.7,
            )
            hypotheses_text = response['choices'][0]['message']['content']
            hypotheses = self.parse_hypotheses(hypotheses_text)
            return hypotheses
        except Exception as e:
            print(f"Error generating hypotheses: {e}")
            return []

    def parse_hypotheses(self, text: str) -> List[str]:
        """
        Parses hypotheses from GPT output.
        """
        try:
            lines = text.strip().split('\n')
            hypotheses = []
            for line in lines:
                if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-')):
                    hypothesis = line.strip().lstrip('0123456789.-) ').strip()
                    if hypothesis:
                        hypotheses.append(hypothesis)
            return hypotheses
        except Exception as e:
            print(f"Error parsing hypotheses: {e}")
            return []

    def generate_code_to_test_hypotheses(self, hypothesis: str) -> str:
        """
        Uses GPT to generate Python code to test a given hypothesis.
        """
        prompt = f"""
You are a data scientist who needs to test the following hypothesis based on a pandas DataFrame called 'df':

Hypothesis:
\"\"\"{hypothesis}\"\"\"

Write Python code to test this hypothesis using appropriate statistical or machine learning techniques. Ensure the code is robust and includes necessary imports. Do not include any explanations, only provide the code.

At the end of your code, store the outcome (True if the hypothesis is supported, False otherwise) in a variable called 'hypothesis_result'. Also, provide any relevant output or results in a variable called 'result' (e.g., plots, statistical test results).
"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                n=1,
                stop=None,
                temperature=0.7,
            )
            code = response['choices'][0]['message']['content']
            return code
        except Exception as e:
            print(f"Error generating code for hypothesis '{hypothesis}': {e}")
            return ""

    def validate_code(self, code: str) -> bool:
        """
        Validates the generated code to ensure it is syntactically correct.
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            print(f"Syntax error in generated code: {e}")
            return False

    def execute_code(self, code: str) -> Dict[str, Any]:
        """
        Safely executes the generated code and captures the output.
        """
        local_vars = {'df': self.df.copy(), 'pd': pd, 'np': np}
        try:
            # Validate code before execution
            if not self.validate_code(code):
                return {'hypothesis_result': None, 'result': None}
            # Limit built-ins and globals for security
            safe_globals = {'__builtins__': {}}
            # Execute the code
            exec(code, safe_globals, local_vars)
            # Retrieve 'hypothesis_result' and 'result' from local_vars
            hypothesis_result = local_vars.get('hypothesis_result', None)
            result = local_vars.get('result', None)
            return {'hypothesis_result': hypothesis_result, 'result': result}
        except Exception as e:
            print(f"Error executing code:\n{code}\nException: {e}\nTraceback: {traceback.format_exc()}")
            return {'hypothesis_result': None, 'result': None}

    def run(self):
        """
        Runs the unsupervised analysis loop.
        """
        stats_json = self.generate_statistics()
        for iteration in range(self.max_iterations):
            print(f"\nIteration {iteration + 1}/{self.max_iterations}")
            hypotheses = self.generate_hypotheses(stats_json)
            if not hypotheses:
                print("No hypotheses generated.")
                break
            for hypothesis in hypotheses:
                print(f"\nTesting Hypothesis: {hypothesis}")
                code = self.generate_code_to_test_hypotheses(hypothesis)
                if not code.strip():
                    print("No code generated to test the hypothesis.")
                    continue
                print(f"\nGenerated Code:\n{code}")
                execution_result = self.execute_code(code)
                hypothesis_result = execution_result.get('hypothesis_result', None)
                result = execution_result.get('result', None)
                self.previous_hypotheses.append({
                    'hypothesis': hypothesis,
                    'worked': 'Yes' if hypothesis_result else 'No'
                })
                if hypothesis_result is not None:
                    print(f"\nHypothesis Result: {'Supported' if hypothesis_result else 'Not Supported'}")
                    if result is not None:
                        print(f"\nAdditional Result:\n{result}")
                else:
                    print("Failed to execute code or 'hypothesis_result' not returned.")
            # Update stats or data if necessary before the next iteration
            # For unsupervised analysis, we may not need to update stats
        print("\nAnalysis complete.")

# Example usage
if __name__ == "__main__":
    # Replace 'YOUR_OPENAI_API_KEY' with your actual OpenAI API key
    openai.api_key = 'YOUR_OPENAI_API_KEY'

    # Sample DataFrame and column descriptions
    data = {
        'age': [25, 30, 22, 40, 35, 28, 45, 50, 27, 33],
        'income': [50000, 60000, 52000, 80000, 75000, 58000, 90000, 100000, 56000, 68000],
        'gender': ['Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'purchased': [0, 1, 0, 1, 1, 0, 1, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    column_descriptions = {
        'age': 'Age of the individual in years.',
        'income': 'Annual income of the individual in USD.',
        'gender': 'Gender of the individual.',
        'purchased': 'Whether the individual made a purchase (1) or not (0).'
    }

    analyzer = GPTUnsupervisedAnalyzer(df, column_descriptions, max_iterations=3)
    analyzer.run()
