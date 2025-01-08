
import pandas as pd
import textwrap
import re
import eli5
# from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
import numpy as np
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


hypotheses_dict = {
    "General Hypothesis": [
        "Hypotheses for dataset analysis explore relationships between variables, differences in group means or proportions, and underlying patterns. These hypotheses help identify key factors, trends, or structural changes in the data, making them essential for statistical testing, model building, and exploratory analysis.",
        "Mean Comparison Hypotheses: Test if the mean of a numerical feature is significantly different between groups.",
        "Proportion Comparison Hypotheses: Check if the proportion of categories is significantly different between groups.",
        "Correlation Hypotheses: Determine if there is a significant correlation between two continuous variables.",
        "Trend Analysis Hypotheses: Test if a time-series dataset shows a significant trend or seasonality.",
        "Regression Hypotheses: Check if a particular variable significantly predicts another variable in a regression model.",
        "Chi-Square Test for Independence: Test if two categorical variables are independent.",
        "ANOVA (Analysis of Variance): Compare the means across multiple groups.",
        "Equality of Variance (Levene’s or Bartlett’s Test): Check if the variance of a numerical variable is the same across groups.",
        "Normality Tests: Test if the distribution of a variable follows a normal distribution.",
        "Homogeneity Tests: Determine if different samples come from populations with the same distribution.",
        "Survival Analysis Hypotheses: Evaluate the time until an event happens and whether factors influence it.",
        "Change Point Analysis: Test if and when a dataset’s mean or variance changes over time.",
        "Equality of Medians (Mann-Whitney U test, Wilcoxon signed-rank): Test if the median of two samples is equal.",
        "Equality of Distribution (Kolmogorov-Smirnov test): Compare the distribution of two independent samples.",
        "Cross-sectional Homogeneity Tests: Test if the dataset has consistent patterns across different sections."
    ],
    "Clustering Hypothesis": [
        "Clustering hypotheses explore the existence of distinct groups within a dataset, where each cluster shares similar characteristics. These hypotheses can reveal whether clusters correspond to specific patterns or risk levels, particularly useful in healthcare, marketing, and other fields where natural grouping provides actionable insights.",
        "Cluster Formation Hypothesis: Use clustering algorithms like k-means, hierarchical clustering, or DBSCAN to identify natural groupings within the data.",
        "Feature Importance within Clusters Hypothesis: Calculate feature importance within clusters to identify sensitive factors impacting outcomes for specific groups.",
        "Cluster Stability Hypothesis: Assess cluster stability by including or excluding specific features and measuring the impact on cluster formation.",
        "Latent Group Hypothesis Using Dimensionality Reduction: Apply PCA or t-SNE to detect latent groups and observe significant differences in key outcomes.",
        "Optimal Number of Clusters Hypothesis: Use techniques like the Elbow Method or Silhouette Analysis to determine the most representative number of clusters for the data.",
        "Cluster-Feature Interaction Hypothesis: Check if feature interactions vary by cluster, revealing distinct effects on outcomes within each group."
    ],
    "Outlier Detection": [
        "Outlier detection methods aim to identify data points that are significantly different from the majority. These hypotheses help detect unusual observations that might indicate noise, data entry errors, or extreme values. Addressing outliers can prevent skewed analysis and improve data accuracy.",
        "Z-Score Test for Outliers: Identify outliers by calculating the z-score of each observation and flagging points with extreme values (e.g., |z| > 3).",
        "Interquartile Range (IQR) Method: Identify outliers by calculating the IQR and flagging values outside the range of Q1 - 1.5*IQR to Q3 + 1.5*IQR.",
        "Mahalanobis Distance: Use the Mahalanobis distance to detect multivariate outliers based on the mean and covariance structure of the dataset.",
        "DBSCAN (Density-Based Spatial Clustering of Applications with Noise): Detect outliers by clustering data points based on density and identifying noise points that do not belong to any cluster."
    ],
    "Anomaly Detection": [
        "Anomaly detection techniques aim to identify unusual patterns or behaviors that deviate significantly from the norm. These hypotheses help detect potentially interesting events, such as fraud or unexpected deviations in data that may require investigation.",
        "Isolation Forest: Apply Isolation Forest to separate anomalies by isolating observations that deviate significantly from the majority.",
        "Change Point Detection (CUSUM, BOCPD): Detect shifts or changes in data distribution using CUSUM or Bayesian Online Change Point Detection (BOCPD) techniques.",
        "Time-Series Anomaly Detection (e.g., Prophet, Seasonal Decomposition): Use time-series anomaly detection models like Prophet or seasonal decomposition to identify unexpected deviations in trends or seasonality.",
        "Chi-Square Test for Categorical Anomalies: Apply the Chi-Square test to detect unusual patterns in categorical variables, identifying anomalous categories with unexpectedly high or low occurrences.",
        "Kolmogorov-Smirnov (K-S) Test for Distributional Anomalies: Use the K-S test to compare the observed and expected distributions, highlighting any significant deviations."
    ],
    "Cause-Effect Relationships": [
        "Cause-effect relationship hypotheses aim to explore potential causative factors between variables. These hypotheses help understand direct, indirect, or conditional effects between variables, revealing complex interactions that can influence outcomes.",
        "Direct Cause-Effect Hypothesis: Test for a direct causal relationship between two variables, assessing if one directly affects the other.",
        "Mediation Hypothesis: Investigate if a third variable mediates the relationship between an independent and dependent variable, indicating an indirect pathway.",
        "Moderation Hypothesis: Assess if the relationship between two variables changes based on the level of a third variable, suggesting a conditional effect.",
        "Indirect Cause-Effect Hypothesis (Indirect Causation): Examine if the effect of one variable on another is mediated by multiple steps or factors, indicating a chain of causation.",
        "Causal Chain Hypothesis: Test a sequence of cause-effect relationships where each variable influences the next in a specified order.",
        "Reciprocal Causation Hypothesis: Analyze if two variables influence each other mutually, creating a feedback loop between them.",
        "Cause-Effect Interaction Hypothesis: Test if the interaction of two variables affects the outcome, revealing complex effects between predictors.",
        "Temporal Causation Hypothesis: Investigate if one variable predicts another over time, considering time-lagged effects on outcomes.",
        "Conditioned Cause-Effect Hypothesis: Test if a cause-effect relationship is conditioned by specific circumstances or thresholds, affecting the impact.",
        "Reverse Causation Hypothesis: Check if the hypothesized outcome variable may actually influence the predictor, reversing the expected direction of causation.",
        "Spurious Causation Hypothesis: Investigate if a relationship between two variables is actually due to an external confounding factor.",
        "Threshold Effect Hypothesis: Test if the effect of a variable on an outcome is only significant when the variable reaches a certain threshold.",
        "Causal Inference with Instrumental Variables Hypothesis: Use instrumental variables to test for causation when direct experimentation is infeasible, controlling for confounders.",
        "Counterfactual Hypothesis: Explore what would happen to an outcome in the absence of a certain factor, creating a counterfactual scenario for causal inference."
    ],
    "Latent Variable Hypotheses": [
        "Latent variable hypotheses explore relationships involving unobserved or hidden variables that help explain observed data structure. These hypotheses assist in understanding complex constructs, revealing deeper relationships that may not be directly measurable.",
        "Existence of Latent Variable Hypothesis: Test for the presence of unobserved variables that influence observed patterns, suggesting an underlying factor.",
        "Latent Variable-Observable Variable Association Hypothesis: Assess if latent variables associate with specific observable variables, clarifying hidden structure.",
        "Factorial Structure Hypothesis: Test if observed variables load onto a smaller number of latent factors, suggesting a factorial structure.",
        "Dimensionality Hypothesis: Explore if the dataset’s variance can be explained by a limited number of latent dimensions, indicating reduced complexity.",
        "Latent Class or Group Hypothesis: Investigate if hidden classes or groups exist within the data, revealing categorical latent structures.",
        "Measurement Invariance Hypothesis: Test if latent variables maintain consistent relationships across different groups, ensuring validity in comparisons.",
        "Predictive Power of Latent Variable Hypothesis: Assess if latent variables can predict specific outcomes, indicating their impact on observed data.",
        "Latent Growth Hypothesis: Analyze if changes in latent variables over time explain longitudinal data trends.",
        "Construct Validity Hypothesis: Test if observed variables accurately measure the latent constructs they are meant to represent.",
        "Latent Variable Mediation Hypothesis: Investigate if latent variables mediate relationships between observed variables, clarifying indirect pathways.",
        "Residualized Latent Variable Hypothesis: Assess if latent variables, after controlling for observable variables, still impact outcomes.",
        "Interaction of Latent Variables Hypothesis: Test if interactions among latent variables affect observed outcomes, indicating interdependent hidden factors.",
        "Latent Trait Distribution Hypothesis: Explore the distribution of latent traits within the population, identifying variance in unobserved characteristics.",
        "Latent Structure Instability Hypothesis: Test if the structure of latent variables shifts over time or across samples, suggesting structural variability.",
        "Effect of External Variable on Latent Structure Hypothesis: Investigate if external variables affect the structure or interpretation of latent variables.",
        "Higher-Order Latent Variable Hypothesis: Test if multiple latent variables combine into a second-order latent structure, adding hierarchical layers."
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
        cols_info_prompt += f'Datatype: {str(X[col].dtype)}\n'

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

def generate_hypothesis_prompt(hypothesis_type, dataset, num_hypothesis = 10):
    dataset_description = textwrap.dedent(dataset.description)
    hypothesis_description = hypotheses_dict[hypothesis_type][0]
    hypothesis_type_prompt = 'Hypothesis Generation Methods:\n' + '\n'.join(hypotheses_dict[hypothesis_type][1:])
    beginning = 'Firstly, I will provide you with some information and then ask you to complete a task based on this information.'
    # instruction_prompt = f"""  
    #                         Dataframe 'dataset' is given. No need to generate the data. 
    #                         Please provide a python code with a list of {num_hypothesis} hypotheses for the dataset described above. For each hypothesis:
    #                         1. Give me business insights based on the hypothesis result.
    #                         2. Write a brief description and save it as a string variable called description.
    #                         3. Provide Python code to test the hypothesis using the dataset DataFrame. The code should write the function test(dataset) to implement appropriate statistical test and return the result as a binary variable called result, which is equal to 1 if we accept the hypothesis. All imports and helper functions shall be inside test() function. Do not rename test function and description variable.
    #                         Return the response using the following format: 
    #                         '''python
    #                         # Hypothesis 1. 
    #                         insights = "[specific business insights for hypothesis 1]"
    #                         description = "[description for hypothesis 1]" 
    #                         [test() for hypothesis 1]
    #                         # Hypothesis 2.
    #                         insights = "[specific business insights for hypothesis 2]"
    #                         description = "[description for hypothesis 2]" 
    #                         [test() for hypothesis 2]
    #                         …
    #                         # Hypothesis {num_hypothesis}.
    #                         insights = "[specific business insights for hypothesis {num_hypothesis}]"
    #                         description = "[description for hypothesis {num_hypothesis}]" 
    #                         [test() for hypothesis {num_hypothesis}]
    #                     """
    instruction_prompt = f"""  
                                Dataframe 'dataset' is given. No need to generate the data. 
                                Please provide a python code with a list of {num_hypothesis} hypotheses for the dataset described above. For each hypothesis:
                                1. Give me business insights based on the {hypothesis_type}. 
                                2. Write a brief description based on the previous insights and save it as a string variable called description.
                                3. Provide detailed step by step chain-of-thought and save it to variable COT.
                                4. Provide Python code based on COT variable to test the hypothesis using the dataset DataFrame. The code should write the function test(dataset) to implement appropriate statistical test and return the result as a binary variable called result, which is equal to 1 if we accept the hypothesis. All imports and helper functions shall be inside test() function. For any hyperparameters choose values based on dataset statistics (e.g. varience, dataset shape)Do not rename test function and description variable. 
                                Return the response using the following format: 
                                '''python
                                # Hypothesis 1. 
                                insights = "[specific business insights for hypothesis 1]"
                                description = "[description for hypothesis 1]" 
                                COT = "[Detailed chain-of-thought for hypothesis 1: Step-by-step thought process to identify the hypothesis requirements, select the correct statistical test, structure the function, and interpret the results.]"
                                [test() for hypothesis 1]
                                # Hypothesis 2.
                                insights = "[specific business insights for hypothesis 2]"
                                description = "[description for hypothesis 2]" 
                                COT = "[Detailed chain-of-thought for hypothesis 2: Step-by-step thought process to identify the hypothesis requirements, select the correct statistical test, structure the function, and interpret the results.]"
                                [test() for hypothesis 2]
                                …
                                # Hypothesis {num_hypothesis}.
                                insights = "[specific business insights for hypothesis {num_hypothesis}]"
                                description = "[description for hypothesis {num_hypothesis}]" 
                                COT = "[Detailed chain-of-thought for hypothesis {num_hypothesis}: Step-by-step thought process to identify the hypothesis requirements, select the correct statistical test, structure the function, and interpret the results.]"
                                [test() for hypothesis {num_hypothesis}]
                                '''
                            """

    cols_info = get_cols_info_prompt(dataset)
    final_prompt = (beginning + '\n' + strip_multiline_string(dataset_description) + '\n' + 
                    strip_multiline_string(cols_info) + '\n' + 
                    strip_multiline_string(hypothesis_description) + '\n' + 
                    strip_multiline_string(hypothesis_type_prompt) + '\n' + 
                    strip_multiline_string(instruction_prompt))
    return final_prompt 


def generate_prompt(instruction_type, dataset, transformation_type=None, hypothesis=None):
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
                            Important note: generate not less than 50 features, strictly follow the format. Instead of loops use comprehensions on both data and feature description.
                           
                         """.strip(),
        'engineering'   : """ 
                            Please generate 50 new features based on the given features.
                            Your response should follow the format below, where transformation function is some operation you do over existing features. 
                            Do not use for loop. Do not use lambda functions. Do not use features other than given above. 
                            Do not name features like 'feature_1'. Give meaningfull name that describes transformation.
                            Do not use target feature. Do not create/read 'data' and 'features_description' objects.
                            Use feature names exactly as they are given above. New features must be of numerical datatype. 
                            Important note: generate not less than 50 features, strictly follow the format. Instead of loops use comprehensions on both data and feature description.
                            
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
                                      
            .............
                                      
            #49.Feature 49.
            features_description['new_feature_49'] = 'description of new_feature_49'
            data['new_feature_49'] = transformation(data['old_feature_49'])
                                      
            #50.Feature 50.
            features_description['new_feature_50'] = 'description of new_feature_50'
            data['new_feature_50'] = transformation(data['old_feature_50'])

            
            """)
    
    # final prompt
    if hypothesis:
        prompt = (
                strip_multiline_string(dataset_description) + '\n' + 
                '#############################################################' + '\n' + 
                
                cols_info + 
                
                '#############################################################' + '\n' +
                 
                 f'List of hypothesis already tested for this dataset:\n{hypothesis}' + '\n' +
                
                '#############################################################' + '\n' + #cols_stats 
                strip_multiline_string(instructions[instruction_type]) +
                strip_multiline_string(transformation_prompt) +
                strip_multiline_string(format_prompt) + function_scheme
        )
    else:
        prompt = (
                strip_multiline_string(dataset_description) + '\n' + 

                '#############################################################' + '\n' + 

                cols_info + '\n' + #cols_stats + 

                '#############################################################'+ '\n' + 

                strip_multiline_string(instructions[instruction_type]) +
                strip_multiline_string(transformation_prompt) +
                strip_multiline_string(format_prompt) + function_scheme
        )
    return prompt


def extract_python_code_hyp(gpt_response):
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



def hyp_results_to_text(hypthosis_results, llm_model):

    prompt = f""" 
                I have the following list:
                {hypthosis_results}
                Where first element of tuple is  a hypothesis and second element is truthness of hypothesis. 
                1 - if hypothesis is true, 0 - otherwise.
                Rewrite each hypothesis as a text. 
             """
    prompt = strip_multiline_string(prompt)
    llm_output = llm_model.llm_call(prompt)
    return llm_output

def hyp_res_to_text(hypothesis_results):
    res = ''
    for tup in hypothesis_results:
        tmp = tup[0]+': ' + str(bool(tup[1]))
        res += (tmp + '\n')
    return res



def auto_ml_agent(
    dataset_name,
    ml_model_type,
    llm_model_type,
    model_name,
    num_of_generations=1,
    num_of_features=20,
    debug=False,
    output_file='df_for_experiments.csv',
    hypothesis=None
):
  
    dataset = DataReader(dataset_name)
    ml_model = Model(ml_model_type, 'classification')
    llm_model = LLM(llm_model_type)

    if debug:
        print(f'Chosen columns: {dataset.chosen_features}\n')

    for i in tqdm(range(num_of_generations)):
        print(f'GENERATION {i + 1}:')
        
        # Preprocessing
        if i == 0:
            preprocessing_prompts = [
                ('Encoding Data', 'prep_1_prompt'),
                ('String Features Preprocessing', 'prep_2_prompt'),
                ('Datetime Features Preprocessing', 'prep_3_prompt'),
            ]
            
            for transformation, var_name in preprocessing_prompts:
                prompt = generate_prompt(
                    instruction_type='preprocessing',
                    dataset=dataset,
                    transformation_type=transformation,
                    hypothesis=hypothesis
                )
                print(prompt)
                output = llm_model.llm_call(prompt, model_name)
                code = extract_python_code(output)
                generate_features(dataset=dataset, code_text=code)
                
                if debug:
                    print(f'After {transformation}: {dataset.train_input.shape}')

        additional_preprocessing = [
            ('Data Cleaning', 'prep_4_prompt'),
            ('Dimensionality Reduction', 'prep_5_prompt'),
        ]
        
        for transformation, var_name in additional_preprocessing:
            prompt = generate_prompt(
                instruction_type='preprocessing',
                dataset=dataset,
                transformation_type=transformation,
                hypothesis=hypothesis
            )
            print(prompt)
            output = llm_model.llm_call(prompt, model_name)
            code = extract_python_code(output)
            generate_features(dataset=dataset, code_text=code)
            
            if debug:
                print(f'After {transformation}: {dataset.train_input.shape}')

        prepare_data_for_model(dataset=dataset)
        ml_model.fit(data=dataset)
        
        if debug:
            print(f'Model score after gen {i} (prep): {ml_model.score}')

        feature_importances = get_feature_importance(ml_model)
        select_features(dataset=dataset, feature_importances=feature_importances, num_features=num_of_features, temp=1)
        dataset.chosen_features = list(dataset.train_input_selected.columns)

        # Feature Engineering
        engineering_prompts = [
            ('Feature Scaling', 'eng_1_prompt'),
            ('Cross Feature Engineering', 'eng_2_prompt'),
            ('Dimensionality Reduction', 'eng_3_prompt'),
        ]
        
        for transformation, var_name in engineering_prompts:
            prompt = generate_prompt(
                instruction_type='engineering',
                dataset=dataset,
                transformation_type=transformation
            )
            output = llm_model.llm_call(prompt, model_name)
            code = extract_python_code(output)
            generate_features(dataset=dataset, code_text=code)
            
            if debug:
                print(f'After {transformation}: {dataset.train_input.shape}')

        prepare_data_for_model(dataset=dataset)
        ml_model.fit(data=dataset)
        
        if debug:
            print(f'Model score after gen {i} (eng): {ml_model.score}')

        feature_importances = get_feature_importance(ml_model)
        if i + 1 == num_of_generations:
            select_features(
                dataset=dataset,
                feature_importances=feature_importances,
                num_features=num_of_features,
                temp=1,
                method='positive importance'
            )
        else:
            select_features(
                dataset=dataset,
                feature_importances=feature_importances,
                num_features=num_of_features,
                temp=1
            )

        dataset.chosen_features = list(dataset.train_input_selected.columns)
        ml_model.fit(data=dataset, subset='top')
        
        if debug:
            print(f'Model score with selected features: {ml_model.score}')

    
    df = dataset.train_input_clean.copy()
    df['Survived'] = dataset.train_labels
    df.to_csv(output_file)

    print('RESULTS OF AUTO ML AGENT IN THE END:')