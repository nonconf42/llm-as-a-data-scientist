
import pandas as pd
import textwrap
import re
import eli5
# from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
import numpy as np

instruction_dict = {
    'preprocessing' : 'feature_preprocessing',
    'emgineering' : 'generate_features'
}

clean_instructions = {
        'preprocessing' : """
                            Please generate 50 new features based on the preprocessing of the given features. Do not use features other than given above.
                            Your response should follow the format below, where transformation function is some preprocessing you do over one existing feature. 
                            Transformation should not include interactions between several features. 
                            Do not use for loop. Do not use features other than given above.
                            Do not use target feature. Do not create/read 'data' and 'features_description' objects.
                            Use feature names exactly as they are given above. New features must be of numerical datatype.
                            Format:
                         """.strip(),
        'engineering'   : """ 
                            Please generate 50 new features based on the given features.
                            Your response should follow the format below, where transformation function is some operation you do over existing features. 
                            Do not use for loop. Do not use features other than given above. 
                            Do not use target feature. Do not create/read 'data' and 'features_description' objects.
                            Use feature names exactly as they are given above. New features must be of numerical datatype. 
                            Format:
                          """.strip()
        }


preprocessing_techniques = {
     "Feature Scaling": [
        "Normalization",
        "Clipping",
        "Logarithmic Transformation",
        "Binning/Discretization",
        "Aggregations",
        "Fourier Transforms",
        "Polynomial Features"
    ],
    
    "Cross Feature Engineering": [
        "Ratios and Differences",
        "Interaction Features",
        "Cross-Feature Creation"
    ],
    
    "Domain-Specific Feature Engineering": [
        "Distances",
        "Clustering",
        "Financial, Health, etc. features"
    ],
    
    "Dimensionality Reduction": [
        "PCA",
        "SVD",
        "t-SNE",
        "Autoencoders",
        "UMAP",
        "LDA (Linear Discriminant Analysis)"
    ],
    
    "Data Cleaning": [
        "Outlier Detection and Handling",
        "Noise Reduction",
        "Imputation of Missing Data",
        "Handling Imbalanced Data"
    ],
    
    "Datetime Features Preprocessing": [
        "Extracting Date/Time Components",
        "Rolling/Averaging Windows",
        "Time since event",
        "Lag Features",
        "Time-series Stationarization",
        "Time-series Decomposition"
    ],
    
    "String Features Preprocessing": [
        "Tokenization",
        "Bag of Words",
        "Term Frequency-Inverse Document Frequency",
        "N-grams",
        "Sentiment Analysis",
        "Named Entity Recognition",
        "Word Embeddings"
    ],
    
    "Encoding Data": [
        "Frequency Encoding",
        "Ordinal Encoding",
        "Grouping Rare Categories",
        "One-Hot Encoding",
        "Target Encoding",
        "Binary Encoding",
        "Label Encoding"
    ]
}

def softmax(logits, temperature=1.0):
    """
    Computes the softmax of a list or numpy array of logits with temperature scaling.

    Parameters:
    logits (array-like): Input data (logits) to softmax function.
    temperature (float): Temperature parameter for scaling logits.

    Returns:
    numpy.ndarray: Probability distribution after applying temperature softmax.
    """
    # Convert logits to a NumPy array
    logits = np.asarray(logits, dtype=np.float64)

    # Handle temperature scaling
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    # Scale logits by temperature
    scaled_logits = logits / temperature

    # Numerical stability: subtract the max to prevent overflow
    max_logits = np.max(scaled_logits)
    stabilized_logits = scaled_logits - max_logits

    exp_logits = np.exp(stabilized_logits)
    sum_exp_logits = np.sum(exp_logits)

    # Avoid division by zero
    if sum_exp_logits == 0:
        raise ValueError("Sum of exponential logits is zero. Check input values or temperature.")

    softmax_probs = exp_logits / sum_exp_logits

    return softmax_probs


def normalize_indentation(code_string, indent_size=4):
    lines = code_string.splitlines()

    cleaned_lines = [line.rstrip() for line in lines]

    indent_level = 0
    formatted_code = []

    for line in cleaned_lines:
        stripped = line.lstrip()  
        if stripped:  
            formatted_code.append(' ' * (indent_size * indent_level) + stripped)
        
        if stripped.endswith(":"):  
            indent_level += 1
        elif stripped and stripped in ["return", "pass", "break", "continue"]:
            indent_level = max(0, indent_level - 1)

    return "\n".join(formatted_code)

def extract_python_code(text: str) -> str:
    # Regular expression to find Python code block within triple backticks
    pattern = r"```python(.*?)```"
    
    # Search for the pattern
    code_blocks = re.findall(pattern, text, re.DOTALL)
    
    # Join the code blocks in case there are multiple and return as a string
    return '\n\n'.join(code_blocks).strip()

def strip_multiline_string(s: str) -> str:
    return "\n".join([line.strip() for line in s.splitlines()])


def generate_features(dataset, code_text):
    executables = code_text.split('\n')
    print("before feature gen", dataset.train_input.shape)

    

    data = dataset.train_input.copy()
    # Execute the generated code safely
    # Define a dictionary to use as the local namespace for exec
    local_namespace = {'data': data.copy(), 'pd': pd, 'np': np}

    features_description = {}
    for code in executables:
        print("Code:", code)
        try:
            exec(normalize_indentation(code), globals(), locals())
        except Exception as e:
            print(code)
            print("ERROR:", e)
            pass
    
    # dataset.train_input = data
    for feature in features_description:
        if feature not in data.columns:
            continue
        if feature in dataset.train_input.columns:
            pass # rename feature
        dataset.train_input[feature] = data[feature]
        dataset.features_description[feature] = features_description[feature]
    print("after feature gen", dataset.train_input.shape)






def get_feature_importance(ml_model):
    features = eli5.explain_weights(ml_model.model, top=-1).feature_importances.importances
    feature_importance = {feature.feature : feature.weight for feature in features}
    return feature_importance
    

def prepare_data_for_model(dataset):
    #remove all non-numeric columns from data.train_input
    scaler = StandardScaler()
    dataset.train_input_clean = dataset.train_input._get_numeric_data()
    dataset.train_input_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset.train_input_clean = dataset.train_input_clean.select_dtypes(exclude=[complex])
    dataset.train_input_clean = pd.DataFrame(scaler.fit_transform(dataset.train_input_clean), columns=dataset.train_input_clean.columns)
    dataset.train_input_clean = dataset.train_input_clean.T.drop_duplicates().T
    print("after feature clean", dataset.train_input_clean.shape)

def combine_datasets(dataset):
    # dataset.train_input_new_features ONLY new features from llm
    # dataset.train_input = dataset.train_input + dataset.train_input_new_features
    pass

def select_features(dataset, feature_importances, num_features, temp, method='random'):
    
    # given features and their importances
    # create prob distrib over features - softmax from numpy
    # then select randonly num_features - given prob distrib
    # dataset.train_input_selected = dataset.train_input[[selected_cols]]\
    if method == 'random':
        probabilities = softmax(list(feature_importances.values()), temperature=temp)
        features = list(feature_importances.keys())
        selected_features = np.random.choice(features, num_features, p=probabilities, replace=False)
        dataset.train_input_selected = dataset.train_input[selected_features]

    elif method == 'top':
        top_features = list(feature_importances.keys())[0:num_features]
        dataset.train_input_selected = dataset.train_input[top_features]

    elif method == 'positive importance':
        positive_importance_features = [k for k,v in feature_importances.items() if v > 0]
        dataset.train_input_selected = dataset.train_input[positive_importance_features]

def get_cols_info_prompt(dataset):
    cols_info_prompt = ''
    for k,v in dataset.features_description.items():
        if k in dataset.chosen_features:
            cols_info_prompt += f'{k} ({v})\n'
    return cols_info_prompt

def get_cols_stats_prompt(dataset):
    cols_info_prompt = ''
    X = dataset.train_input
    for col in dataset.chosen_features:
        stats = X[col].describe()
        examples = list(X[col].sample(n=5, replace=False))
        cols_info_prompt += f'Feature {col}:\n'
        cols_info_prompt += f'Count of NaN values: {X[col].isna().sum()}\n'
        cols_info_prompt += f'Datatype: {str(X[col].dtype)}'

        if pd.api.types.is_numeric_dtype(X[col]):
            cols_info_prompt += f'max: {stats["max"]}\n'
            cols_info_prompt += f'min: {stats["min"]}\n'
            cols_info_prompt += f'mean: {stats["mean"]}\n'
            cols_info_prompt += f'std: {stats["std"]}\n'
            
        else:
            cols_info_prompt += f'count of unique values: {stats["unique"]}\n' 
            cols_info_prompt += f'the most frequent value: {stats["top"]}\n'
            cols_info_prompt += f'frequency of mode: {stats["freq"]}\n'

        cols_info_prompt += f'Examples: {examples}\n'
    return cols_info_prompt

def generate_prompt(instruction_type, dataset, transformation_type=None):
    # prompt = problem_desctiption + cols_info + instruction + function_scheme
    # cols only from data.train_input_selected
    # cols_info : cols_types + cols_examples + cols_percentiles + cols_percent_missing + cols_unique_vals

    #dataset description prompt
    dataset_description = textwrap.dedent(dataset.description)
    
    #columns infromation prompt
    cols_stats = get_cols_stats_prompt(dataset)
    cols_info = get_cols_info_prompt(dataset)
    temp = ','.join(preprocessing_techniques[transformation_type])
    transformation_prompt = f'Focus only on {transformation_type} transformations, such as: {temp}.\n'

    # get instruction prompt
    format_prompt = '''\nFormat:One sentence description description of the common idea of the new features.
                       Python code: '''
    instructions = {
        'preprocessing' : """
                            Please generate 50 new features based on the preprocessing of the given features. 
                            Do not use features other than given above.
                            Your response should follow the format below, 
                            where transformation function is some preprocessing you do over one existing feature. 
                            Transformation should not include interactions between several features. Do not name features like 'feature_1'.
                            Give meaningfull name that describes transformation.
                            Do not use for loop. Do not use lambda functions. Do not use features other than given above.
                            Do not use target feature. Do not create/read 'data' and 'features_description' objects.
                            Use feature names exactly as they are given above. New features must be of numerical datatype.
                            Format:
                         """.strip(),
        'engineering'   : """ 
                            Please generate 50 new features based on the given features.
                            Your response should follow the format below, where transformation function is some operation you do over existing features. 
                            Do not use for loop. Do not use lambda functions. Do not use features other than given above. 
                            Do not name features like 'feature_1'. Give meaningfull name that describes transformation.
                            Do not use target feature. Do not create/read 'data' and 'features_description' objects.
                            Use feature names exactly as they are given above. New features must be of numerical datatype. 
                            Format:
                          """.strip()
        }
    
    # get function scheme
    # function_scheme = textwrap.dedent(f"""
    #             def {instruction_dict[instruction_type]}(data: pd.DataFrame) -> pd.DataFrame:
    #             \t#1. new_feature_1 description.
    #             \tdata['new_feature_name_1'] = transformation(data['old_feature_1'])

    #             \t#2. new_feature_2 description.
    #             \tdata['new_feature_name_2'] = transformation(data['old_feature_2'])
                                                
    #             \t#Same logic applies to other N features
    #             \treturn data
    #         """)
    function_scheme = textwrap.dedent(f"""
            #1. Feature 1.
            features_description['new_feature_1'] = 'description of new_feature_1'
            data['new_feature_1'] = transformation(data['old_feature_1'])

            #2. Feature 2.
            features_description['new_feature_2'] = 'description of new_feature_2'
            data['new_feature_2'] = transformation(data['old_feature_2'])

            
            """)
    
    # final prompt
    prompt = (
             strip_multiline_string(dataset_description) + 
             cols_info + '\n' + cols_stats +
             strip_multiline_string(instructions[instruction_type]) +
             strip_multiline_string(transformation_prompt) +
             strip_multiline_string(format_prompt) + function_scheme
    )
    return prompt

    