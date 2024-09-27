import pandas as pd
from datasets.data import DataReader
from models.model import Model
from llms.llm import LLM
from models.utils import *
import argparse
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

NUM_OF_GENERATIONS = 3
NUM_OF_FEATURES = 20

dataset = DataReader('Cars')

ml_model = Model('XGB', 'regression')
llm_model = LLM('gpt')
model_name = "gpt-4o"

preprocessing_prompt = generate_prompt(
    instruction_type='preprocessing',
    dataset=dataset,
    transformation_type='Data Cleaning'
)

#print(preprocessing_prompt)

llm_output = llm_model.llm_call(preprocessing_prompt, model_name)
llm_code = extract_python_code(llm_output)
print(llm_code)
generate_features(dataset=dataset, code_text=llm_code)
prepare_data_for_model(dataset=dataset)
ml_model.fit(data=dataset)
print(f'Model score after Data Cleaning : {ml_model.score}')



# parser = argparse.ArgumentParser(description="Control debug mode")
# parser.add_argument('--debug', action='store_true', help='Enable debug mode')

# args = parser.parse_args()

# if args.debug:
#     print(f'Chosen columns: {dataset.chosen_features}\n')



# for i in tqdm(range(NUM_OF_GENERATIONS)):
#     print(f'GENERATION {i+1}:')
#     ### PREPROCESSING PART ###

#     prep_1_prompt = generate_prompt(
#         instruction_type='preprocessing',
#         dataset=dataset,
#         transformation_type='Data Cleaning'
#     )

#     prep_2_prompt = generate_prompt(
#         instruction_type='preprocessing',
#         dataset=dataset,
#         transformation_type='Encoding Data'
#     )

#     prep_3_prompt = generate_prompt(
#         instruction_type='preprocessing',
#         dataset=dataset,
#         transformation_type='String Features Preprocessing'
#     )

#     llm_prep_1_output = llm_model.llm_call(prep_1_prompt, model_name)
#     llm_prep_2_output = llm_model.llm_call(prep_2_prompt, model_name)
#     llm_prep_3_output = llm_model.llm_call(prep_3_prompt, model_name)

    
#     llm_prep_1_code = extract_python_code(llm_prep_1_output)
#     llm_prep_2_code = extract_python_code(llm_prep_2_output)
#     llm_prep_3_code = extract_python_code(llm_prep_3_output)


#     generate_features(dataset=dataset, code_text=llm_prep_1_code)
#     if args.debug:
#         print(f'After first preprocessing:{dataset.train_input.shape}')

#     generate_features(dataset=dataset, code_text=llm_prep_2_code)
#     if args.debug:
#         print(f'After second preprocessing:{dataset.train_input.shape}')

#     generate_features(dataset=dataset, code_text=llm_prep_3_code)
#     if args.debug:
#         print(f'After third preprocessing:{dataset.train_input.shape}')

#     prepare_data_for_model(dataset=dataset)
#     ml_model.fit(data=dataset)
#     if args.debug:
#         print(f'Model score after gen {i} (prep): {ml_model.score}')

#     feature_importances = get_feature_importance(ml_model)
#     # if args.debug:
#     #     print(f'Feature Importances:\n {feature_importances}')
    
    
#     select_features(dataset, feature_importances, NUM_OF_FEATURES)
#     dataset.chosen_features = list(dataset.train_input_selected.columns)
#     # if args.debug:
#     #     print(dataset.train_input_selected.columns)
     

#     ### ENGINEETING PART ###
        
#     eng_1_prompt = generate_prompt(
#         instruction_type='engineering',
#         dataset=dataset,
#         transformation_type='Feature Scaling'
#     )
    

#     eng_2_prompt = generate_prompt(
#         instruction_type='engineering',
#         dataset=dataset,
#         transformation_type='Cross Feature Engineering'
#     )

#     eng_3_prompt = generate_prompt(
#         instruction_type='engineering',
#         dataset=dataset,
#         transformation_type='Dimensionality Reduction'
#     )

#     llm_eng_1_output = llm_model.llm_call(eng_1_prompt, model_name)
#     llm_eng_2_output = llm_model.llm_call(eng_2_prompt, model_name)
#     llm_eng_3_output = llm_model.llm_call(eng_3_prompt, model_name)

    
#     llm_eng_1_code = extract_python_code(llm_eng_1_output)
#     llm_eng_2_code = extract_python_code(llm_eng_2_output)
#     llm_eng_3_code = extract_python_code(llm_eng_3_output)

#     generate_features(dataset=dataset, code_text=llm_eng_1_code)
#     if args.debug:
#         print(f'After first engineering:{dataset.train_input.shape}')

#     generate_features(dataset=dataset, code_text=llm_eng_2_code)
#     if args.debug:
#         print(f'After second engineering :{dataset.train_input.shape}')

#     generate_features(dataset=dataset, code_text=llm_eng_3_code)
#     if args.debug:
#         print(f'After third engineering:{dataset.train_input.shape}')

#     prepare_data_for_model(dataset=dataset)
#     ml_model.fit(data=dataset)
#     if args.debug:
#         print(f'Model score after gen {i} (eng): {ml_model.score}')

#     feature_importances = get_feature_importance(ml_model)
#     # if args.debug:
#     #     print(f'Feature Importances:\n {feature_importances}')
    
#     if i + 1 == NUM_OF_GENERATIONS:
#         select_features(dataset, feature_importances, NUM_OF_FEATURES, method='top')
#         dataset.chosen_features = list(dataset.train_input_selected.columns)
#     else:
#         select_features(dataset, feature_importances, NUM_OF_FEATURES)
#         dataset.chosen_features = list(dataset.train_input_selected.columns)
        
#     if args.debug:
#         print(dataset.train_input_selected.columns)

#     ml_model.fit(data=dataset, subset='top')
#     if args.debug:
#         print(f'Model score with selected features: {ml_model.score}')
