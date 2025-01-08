import pandas as pd
from datasets.data import DataReader
# from models.model import Model
from llms.llm import LLM
from models.utils import *
import openai
import os
import re
import tempfile
import inspect
import pandas as pd
from datasets.data import DataReader
from models.model import Model
from llms.llm import LLM
from models.utils import *
import argparse
import warnings
from tqdm import tqdm
from clf_fit_agent import *
warnings.filterwarnings("ignore")

def extract_python_code(gpt_response):
    # Regular expression to capture code between triple backticks labeled as python
    code_blocks = re.findall(r"```python(.*?)```", gpt_response, re.DOTALL)
    
    # If there are multiple code blocks, join them together
    extracted_code = "\n\n".join(code_blocks)
    
    return extracted_code.strip()

def extract_hypotheses_text(response):
    """
    Extracts the full text of each hypothesis from the response string.
    
    Parameters:
        response (str): The response text containing multiple hypotheses and their functions.
    
    Returns:
        list: A list where each element is a string containing the full text of a hypothesis.
    """
    # Regular expression to match each hypothesis section
    pattern = r"# Hypothesis \d+\..*?(?=# Hypothesis|\Z)"
    
    # Find all matches of the pattern in the response
    matches = re.findall(pattern, response, re.DOTALL)
    
    # Return list of hypothesis text blocks
    return [match.strip() for match in matches]
# import argparse
# import warnings
# from tqdm import tqdm
# from clf_fit_agent import *
# warnings.filterwarnings("ignore")






import inspect
def hyp_results_to_text(hypthosis_results, llm_model):

    prompt = f""" 
                I fave the following list:
                {hypthosis_results}
                Where first element of tuple is  a hypothesis and second element is truthness of hypothesis. 
                1 - if hypothesis is true, 0 - otherwise.
                Rewrite each hypothesis as a text. Start every new hypothesis from new line.
             """
    prompt = strip_multiline_string(prompt)
    llm_output = llm_model.llm_call(prompt)
    return llm_output

# def run_hypothesis_tests(llm_model, dataset, hypothesis_type):
#     prompt = generate_hypothesis_prompt(hypothesis_type=hypothesis_type, dataset=dataset, num_hypothesis=10)
#     print(prompt)
#     llm_output = llm_model.llm_call(prompt)
#     print(llm_output)
#     python_code = extract_python_code(llm_output)
#     print('###################')
#     print(python_code)
#     print('###################')
#     hypothesis_list = extract_hypotheses_text(python_code)
#     print(hypothesis_list)
#     imports = '''
#                 # Data manipulation and analysis
#                 from scipy.stats import chi2_contingency
#                 import scipy.stats
#                 import numpy as np
#                 import pandas as pd
#                 # Machine learning models and preprocessing
#                 from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
#                 from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
#                 from sklearn.pipeline import Pipeline
#                 # Commonly used machine learning algorithms
#                 from sklearn.linear_model import LogisticRegression, LinearRegression
#                 from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
#                 from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
#                 from sklearn.svm import SVC, SVR
#                 from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
#                 from sklearn.naive_bayes import GaussianNB
#                 from sklearn.cluster import KMeans
#                 # Model evaluation metrics
#                 from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, mean_squared_error, r2_score
#                 # Dimensionality reduction
#                 from sklearn.decomposition import PCA
#               '''
#     imports = strip_multiline_string(imports)
    
#     results = []  
#     for code in hypothesis_list:
        
#         full_code = imports + '\n' + code
#         print(full_code)
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.py') as temp_module:
#             temp_module_name = os.path.basename(temp_module.name).split('.')[0]
#             temp_module.write(full_code.encode('utf-8'))
#             temp_module_path = temp_module.name

#             try:
#                 spec = importlib.util.spec_from_file_location(temp_module_name, temp_module_path)
#                 module = importlib.util.module_from_spec(spec)
#                 module.dataset = dataset.train_base 
#                 print('%%%%%%%%%%%%%%%%%%%%%%')
#                 print(f'###################\n{dir(module)}')
#                 print(inspect.getsource(module))
#                 if hasattr(module, 'test'):

#                     result = module.test(dataset.train_base)
#                     print(f'result::::{result}')
#                     description = module.description
#                     results.append((description, result))
#             except Exception as e:
#                 print('ERROR')
#                 print(e)
#                 print(full_code)
#                 print('#####')
#             finally:
#                 os.remove(temp_module_path)
    
#     return results



import tempfile
import importlib.util
import os
import inspect




# Set up your OpenAI API key
api_key = "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"  # Replace with your actual API key
llm_model = LLM('gpt')
model_name = "gpt-4o"

hypothesis_results = []

df = DataReader('Titanic')

h_list = [
    "General Hypothesis",
    "Clustering Hypothesis",
    "Outlier Detection",
    "Anomaly Detection",
    "Cause-Effect Relationships",
    "Latent Variable Hypotheses"
]

for hyp_type in h_list:
    prompt = generate_hypothesis_prompt(hypothesis_type=hyp_type, dataset=df,num_hypothesis=10)
    llm_output = llm_model.llm_call(prompt)
    python_code = extract_python_code(llm_output)
    hypothesis_list = extract_hypotheses_text(python_code)
    dataset = df.train_base
    imports = ''' 
                    # Data manipulation and analysis
                    from scipy.stats import chi2_contingency
                    import scipy.stats
                    import numpy as np
                    import pandas as pd
                    # Machine learning models and preprocessing
                    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
                    from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
                    from sklearn.pipeline import Pipeline
                    # Commonly used machine learning algorithms
                    from sklearn.linear_model import LogisticRegression, LinearRegression
                    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
                    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
                    from sklearn.svm import SVC, SVR
                    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
                    from sklearn.naive_bayes import GaussianNB
                    from sklearn.cluster import KMeans
                    # Model evaluation metrics
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, mean_squared_error, r2_score
                    # Dimensionality reduction
                    from sklearn.decomposition import PCA
                '''
    imports = strip_multiline_string(imports)
    for code in hypothesis_list:
        code = imports + '\n' + code 
        with tempfile.NamedTemporaryFile(delete=False, suffix='.py') as temp_module:
            temp_module_name = os.path.basename(temp_module.name).split('.')[0]
            temp_module.write(code.encode('utf-8'))
            temp_module_path = temp_module.name
        try:
            # Import the module
            import importlib.util
            spec = importlib.util.spec_from_file_location(temp_module_name, temp_module_path)
            module = importlib.util.module_from_spec(spec)
            module.dataset = dataset
            spec.loader.exec_module(module)
            #print(f'##########\n {dir(module)}')
            if hasattr(module, 'test'):
                result = module.test(dataset)
                description = module.description
                hypothesis_results.append((description, result))
                print(module.description)
                print(f'INSIGHTS: {module.insights}')
                print(result)
                print('#####')
                print(f'COT:{module.COT}')
        except Exception as e:
            print('ERROR')
            print(e)
            print(code)
            print('#####')
        finally:
            # Clean up the temporary file
            os.remove(temp_module_path)
print('$$$$$$$$$$$$$$$$$$')
print(hypothesis_results)

print('Cleaned hypothesis results:')
cleaned_hypothesis_results = [hyp for hyp in hypothesis_results if hyp[1] != None]
print(cleaned_hypothesis_results)
print(f'Len: {len(cleaned_hypothesis_results)}')
print('$$$$$$$$$$$$$$$$$$$$$$$$')
bag_of_hyp = hyp_results_to_text(cleaned_hypothesis_results, llm_model=llm_model)





