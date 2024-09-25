import pandas as pd

dataset_descriptions = {
    'Titanic' : '''This is titanic dataset. The goal is to predict which passenger survived. While there
                    was some element of luck involved in surviving, it seems some groups of people were
                    more likely to survive than others. Below is the data description in the format - Variable (definition):

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
            

    
