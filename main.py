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

NUM_OF_GENERATIONS = 1
NUM_OF_FEATURES = 20

#temperature_values = [0.1, 1, 10]
#num_of_features_to_select = [10, 20, 50]

dataset = DataReader('Titanic')
ml_model = Model('XGB', 'classification')
llm_model = LLM('gpt')
model_name = "gpt-4o"

parser = argparse.ArgumentParser(description="Control debug mode")
parser.add_argument('--debug', action='store_true', help='Enable debug mode')

args = parser.parse_args()

if args.debug:
    print(f'Chosen columns: {dataset.chosen_features}\n')



for i in tqdm(range(NUM_OF_GENERATIONS)):
    print(f'GENERATION {i+1}:')
    ### PREPROCESSING PART ###
    if i == 0:
        prep_1_prompt = generate_prompt(
            instruction_type='preprocessing',
            dataset=dataset,
            transformation_type='Encoding Data'
        )

        prep_2_prompt = generate_prompt(
            instruction_type='preprocessing',
            dataset=dataset,
            transformation_type='String Features Preprocessing'
        )

        prep_3_prompt = generate_prompt(
            instruction_type='preprocessing',
            dataset=dataset,
            transformation_type='Datetime Features Preprocessing'
        )

        llm_prep_1_output = llm_model.llm_call(prep_1_prompt, model_name)
        llm_prep_2_output = llm_model.llm_call(prep_2_prompt, model_name)
        llm_prep_3_output = llm_model.llm_call(prep_3_prompt, model_name)

        
        llm_prep_1_code = extract_python_code(llm_prep_1_output)
        llm_prep_2_code = extract_python_code(llm_prep_2_output)
        llm_prep_3_code = extract_python_code(llm_prep_3_output)


        generate_features(dataset=dataset, code_text=llm_prep_1_code)
        if args.debug:
            print(f'After first preprocessing:{dataset.train_input.shape}')

        generate_features(dataset=dataset, code_text=llm_prep_2_code)
        if args.debug:
            print(f'After second preprocessing:{dataset.train_input.shape}')

        generate_features(dataset=dataset, code_text=llm_prep_3_code)
        if args.debug:
            print(f'After third preprocessing:{dataset.train_input.shape}')

    prep_4_prompt = generate_prompt(
            instruction_type='preprocessing',
            dataset=dataset,
            transformation_type='Data Cleaning'
        )

    prep_5_prompt = generate_prompt(
            instruction_type='preprocessing',
            dataset=dataset,
            transformation_type='Dimensionality Reduction'
        )
    
    llm_prep_4_output = llm_model.llm_call(prep_4_prompt, model_name)
    llm_prep_5_output = llm_model.llm_call(prep_5_prompt, model_name)
    
    
    llm_prep_4_code = extract_python_code(llm_prep_4_output)
    llm_prep_5_code = extract_python_code(llm_prep_5_output)

    generate_features(dataset=dataset, code_text=llm_prep_4_code)
    if args.debug:
        print(f'After forth preprocessing:{dataset.train_input.shape}')

    generate_features(dataset=dataset, code_text=llm_prep_5_code)
    if args.debug:
        print(f'After fifth preprocessing:{dataset.train_input.shape}')

    
    
    prepare_data_for_model(dataset=dataset)
    ml_model.fit(data=dataset)
    if args.debug:
        print(f'Model score after gen {i} (prep): {ml_model.score}')

    feature_importances = get_feature_importance(ml_model)
    # if args.debug:
    #     print(f'Feature Importances:\n {feature_importances}')
    
    
    select_features(dataset=dataset, feature_importances=feature_importances, num_features=NUM_OF_FEATURES, temp=1)
    dataset.chosen_features = list(dataset.train_input_selected.columns)
    # if args.debug:
    #     print(dataset.train_input_selected.columns)
    

    ### ENGINEETING PART ###
        
    eng_1_prompt = generate_prompt(
        instruction_type='engineering',
        dataset=dataset,
        transformation_type='Feature Scaling'
    )
    

    eng_2_prompt = generate_prompt(
        instruction_type='engineering',
        dataset=dataset,
        transformation_type='Cross Feature Engineering'
    )

    eng_3_prompt = generate_prompt(
        instruction_type='engineering',
        dataset=dataset,
        transformation_type='Dimensionality Reduction'
    )

    llm_eng_1_output = llm_model.llm_call(eng_1_prompt, model_name)
    llm_eng_2_output = llm_model.llm_call(eng_2_prompt, model_name)
    llm_eng_3_output = llm_model.llm_call(eng_3_prompt, model_name)

    
    llm_eng_1_code = extract_python_code(llm_eng_1_output)
    llm_eng_2_code = extract_python_code(llm_eng_2_output)
    llm_eng_3_code = extract_python_code(llm_eng_3_output)

    generate_features(dataset=dataset, code_text=llm_eng_1_code)
    if args.debug:
        print(f'After first engineering:{dataset.train_input.shape}')

    generate_features(dataset=dataset, code_text=llm_eng_2_code)
    if args.debug:
        print(f'After second engineering :{dataset.train_input.shape}')

    generate_features(dataset=dataset, code_text=llm_eng_3_code)
    if args.debug:
        print(f'After third engineering:{dataset.train_input.shape}')

    prepare_data_for_model(dataset=dataset)
    ml_model.fit(data=dataset)
    if args.debug:
        print(f'Model score after gen {i} (eng): {ml_model.score}')

    feature_importances = get_feature_importance(ml_model)
    # if args.debug:
    #     print(f'Feature Importances:\n {feature_importances}')
    
    if i + 1 == NUM_OF_GENERATIONS:
        select_features(dataset=dataset,feature_importances=feature_importances,num_features=NUM_OF_FEATURES,temp=1, method='positive importance')
        dataset.chosen_features = list(dataset.train_input_selected.columns)
    else:
        select_features(dataset=dataset, feature_importances=feature_importances,num_features=NUM_OF_FEATURES, temp=1)
        dataset.chosen_features = list(dataset.train_input_selected.columns)
        
    if args.debug:
        print(dataset.train_input_selected.columns)

    ml_model.fit(data=dataset, subset='top')
    if args.debug:
        print(f'Model score with selected features: {ml_model.score}')
    
    df = dataset.train_input_clean.copy()
    df['Survived'] = dataset.train_labels

    df.to_csv('df_for_experiments.csv')
    print('RESULTS OF AUTO ML AGENT IN THE END:')

    #main_clf(dataset.train_input_clean, dataset.train_labels)


    


# https://www.kaggle.com/code/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy/comments