import pandas as pd

dataset_descriptions = {
    'Titanic' : '''This is titanic dataset. The goal is to predict which passenger survived. While there
                    was some element of luck involved in surviving, it seems some groups of people were
                    more likely to survive than others. Below is the data description in the format - Variable (definition):

                ''',
    'Cars' : '''This is cars dataset. The goal is to predict the price of used cars based on various attributes.
                Below is the data description in the format - Variable (definition):

              '''
}
class DataReader:
    def __init__(self, dataset_name):
        self.train_base = pd.read_csv(f'datasets/{dataset_name}/train.csv')
        self.test_input_base = pd.read_csv(f'datasets/{dataset_name}/test.csv')
        self.description = dataset_descriptions[dataset_name]

        if dataset_name == 'Titanic': 
            self.train_input = self.train_base.drop(['Survived', 'PassengerId'], axis=1)
            self.train_labels = self.train_base['Survived']
            self.features_description = {
                'Pclass' : 'Ticket class: 1 = 1st, 2 = 2nd, 3 = 3rd',
                'Sex' : 'Sex: male, female',
                'Age' : 'Age in years',
                'SibSp' : '# of siblings / spouses aboard the Titanic',
                'Parch' : '# of parents / children aboard the Titanic',
                'Ticket' : 'Ticket number',
                'Fare' : 'Passenger fare',
                'Cabin' : 'Cabin number',
                'Embarked' : 'Port of Embarkation: C = Cherbourg,Q = Queenstown, S = Southampton',\
                'Name' : 'Name of passenger'
            }
            self.chosen_features = [col for col in self.train_base.columns if col not in ['Survived', 'PassengerId']]
            
        elif dataset_name == 'Cars':
            self.train_input = self.train_base.drop(['id', 'price'], axis=1)
            self.train_labels = self.train_base['price']
            self.features_description = {
                'brand': 'The manufacturer or company that produces the vehicle. Crucial for assessing depreciation and technology advancements',
                'model': 'The specific name or version of a car produced by a brand.',
                'model_year': 'The year in which the vehicle was manufactured or introduced.',
                'milage': 'The total distance the car has traveled, usually measured in miles. A key indicator of wear and tear and potential maintenance requirements. ',
                'fuel_type': 'The type of fuel the car uses, such as gasoline, diesel, hybrid or electric.',
                'engine': 'The mechanical component that powers the car, often described by its size and power output.',
                'transmission': 'The system that transmits power from the engine to the wheels, either manual, automatic, or another variant.',
                'ext_col': "The color of the vehicle’s exterior.",
                'int_col': "The color of the vehicle’s interior, including seats and trim.",
                'accident': "Indicates whether the car has been involved in any accidents.",
                'clean_title': "Indicates that the car’s title is free of any legal issues such as salvage or rebuild history."
            }
            self.chosen_features = [col for col in self.train_base.columns if col not in ['id', 'price']]


            

    
