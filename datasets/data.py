import pandas as pd

dataset_descriptions = {
    'Titanic' : '''This is titanic dataset. The goal is to predict which passenger survived. While there
                    was some element of luck involved in surviving, it seems some groups of people were
                    more likely to survive than others. Below is the data description in the format - Variable (definition):

                ''',
    'Cars' : '''This is cars dataset. The goal is to predict the price of used cars based on various attributes.
                Below is the data description in the format - Variable (definition):

             ''',
    'Diamonds' : """This is diamonds dataset. The goal is to predict price of diamonds based on carat, cut, color,
                    clarity, depth, table, x, y ,z. Below is the data description in the format -  Variable (definition):
        
                 """,

    'Smoking' : """This is smoking dataset. The goal is to predict an individual's smoking status based on various
                   health indicators.
                   This is binary classification task. 
                   Below is the data description in the format -  Variable (definition):
        
                 """,
    'Shopping' : """The dataset offers a comprehensive view of consumer shopping trends, aiming to uncover patterns and behaviors in retail purchasing. It contains detailed transactional data 
                    across various product categories, customer demographics, and purchase channels. Key features may include:
                    Transaction Details: Purchase date, transaction value, product category, and payment method.
                    Customer Information: Age group, gender, location, and loyalty status.
                    Shopping Behavior: Frequency of purchases, average spend per transaction, and seasonal trends.
                """,
    'Bank' : """ This dataset presents the financial performance of Deutsche Bank, 
                 including key metrics like operating income, net income, and balance sheet indicators. It spans 800 records over time.
             """,

    'Churn' : """ This data set contains details of a bank's customers and the target variable is a binary variable reflecting the fact whether the customer left the bank (closed his account) or he continues to be a customer."""
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
                'SibSp' : 'number of siblings / spouses aboard the Titanic',
                'Parch' : 'number of parents / children aboard the Titanic',
                'Ticket' : 'Ticket number',
                'Fare' : 'Passenger fare',
                'Cabin' : 'Cabin number',
                'Embarked' : 'Port of Embarkation: C = Cherbourg,Q = Queenstown, S = Southampton',\
                'Name' : 'Name of passenger'
            }
            self.chosen_features = [col for col in self.train_base.columns if col not in ['Survived', 'PassengerId']]
            self.label_name = 'Survived'
            self.task = 'classification'

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
            self.label_name = 'price'
            self.task = 'regression'

        elif dataset_name == 'Diamonds':
            self.train_input = self.train_base.drop(['price'], axis=1)
            self.train_labels = self.train_base['price']
            self.features_description = {
                "carat": "The carat value of the Diamond",
                "cut": "The cut type of the Diamond, it determines the shine ('Ideal', 'Premium', 'Good', 'Very Good', 'Fair')",
                "color": "The color value of the Diamond ('E', 'I', 'J', 'H', 'F', 'G', 'D')",
                "clarity": "The clarity type of the Diamond ('SI2', 'SI1', 'VS1', 'VS2', 'VVS2', 'VVS1', 'I1')",
                "depth": "The depth value of the Diamond",
                "table": "Flat facet on its surface — the large, flat surface facet that you can see when you look at the diamond from above",
                "x": "Width of the diamond",
                "y": "Length of the diamond",
                "z": "Height of the diamond",
                "price": "The price of the Diamond in USD"
            }
            self.chosen_features = [col for col in self.train_base.columns if col not in ['price']]
            self.label_name = 'price'
            self.task = 'regression'

        elif dataset_name == 'Smoking':
            self.train_input = self.train_base.drop(['smoking'], axis=1)
            self.train_labels = self.train_base['smoking']
            self.features_description = {
                "age": "Age of the individual, measured in 5-year gaps",
                "height": "Height of the individual in centimeters",
                "weight": "Weight of the individual in kilograms",
                "waist": "Waist circumference length in centimeters",
                "eyesight(left)": "Eyesight measurement for the left eye",
                "eyesight(right)": "Eyesight measurement for the right eye",
                "hearing(left)": "Hearing ability of the left ear",
                "hearing(right)": "Hearing ability of the right ear",
                "systolic": "Systolic blood pressure measurement",
                "relaxation": "Diastolic blood pressure measurement",
                "fasting blood sugar": "Blood sugar level measured after fasting",
                "Cholesterol": "Total cholesterol level in the blood",
                "triglyceride": "Triglyceride level in the blood",
                "HDL": "High-density lipoprotein cholesterol level",
                "LDL": "Low-density lipoprotein cholesterol level",
                "hemoglobin": "Hemoglobin level in the blood",
                "Urine protein": "Protein level in the urine",
                "serum creatinine": "Creatinine level in the blood serum",
                "AST": "Aspartate aminotransferase (glutamic oxaloacetic transaminase) level",
                "ALT": "Alanine aminotransferase (glutamic pyruvic transaminase) level",
                "Gtp": "Gamma-glutamyl transpeptidase (γ-GTP) level",
                "dental caries": "Indicates the presence of dental cavities",
                "smoking": "Smoking status of the individual"
            }
            self.chosen_features = [col for col in self.train_base.columns if col not in ['smoking']]
            self.label_name = 'smoking'
            self.task = 'classification'

        elif dataset_name == 'Shopping':
            self.train_input = self.train_base
            self.chosen_features = self.train_base.columns
            self.features_description = {
                "Customer ID": "Unique identifier for each customer.",
                "Age": "Age of the customer (numerical).",
                "Gender": "Gender of the customer (e.g., Male, Female).",
                "Item Purchased": "Name of the item purchased.",
                "Category": "Product category (Clothing, Footwear, Outerwear, Accessories).",
                "Purchase Amount (USD)": "Total amount spent on the purchase.",
                "Location": "Geographic location of the customer (city)",
                "Size": "Size of the item purchased (if applicable; e.g., S, M, L, XL).",
                "Color": "Color of the item purchased",
                "Season": "Season of the purchase (e.g., Winter, Summer, Spring, Fall).",
                "Review Rating": "Customer's review rating for the purchase (e.g., 1-5 stars).",
                "Subscription Status": "Indicates whether the customer is a subscriber (e.g., Active, Inactive, None).",
                "Payment Method": "Method used for the payment (e.g., Credit Card, PayPal, Cash, Venmo, Debit Card, Bank Transfer).",
                "Shipping Type": "Type of shipping chosen (Express, Free Shipping, Next Day Air, Standard, 2-Day Shipping,Store Pickup",
                "Discount Applied": "Indicates if a discount was applied (e.g., Yes/No ).",
                "Promo Code Used": "Specifies if a promo code was used during the purchase (e.g., Yes/No).",
                "Previous Purchases": "Total number of previous purchases made by the customer.",
                "Preferred Payment Method": "The payment method most frequently used by the customer.",
                "Frequency of Purchases": "How often the customer makes purchases "
            }
            self.task = 'call to action'
        
        elif dataset_name == 'Bank':
            self.train_input = self.train_base
            self.chosen_features = self.train_base.columns
            self.features_description = {
                "Date": "The date corresponding to each financial record (from January 2015 onwards).",
                "Operating_Income": "The income generated from the bank's core business operations.",
                "Expenses": "Total costs incurred during operations.",
                "Net_Income": "Profit after subtracting expenses from operating income.",
                "Assets": "Total assets owned by the bank (e.g., cash, investments).",
                "Liabilities": "The total debts and obligations owed.",
                "Equity": "Shareholders' equity, representing the net value of assets minus liabilities.",
                "Debt_to_Equity": "A financial ratio showing the proportion of debt compared to equity.",
                "ROA": "A profitability metric calculated as net income divided by total assets.",
                "Revenue": "Total income from all operations and activities.",
                "Cash_Flow": "The net cash generated or used in operations.",
                "Profit_Margin": "A ratio showing the percentage of revenue that remains as profit.",
                "Interest_Expense": "Costs associated with the bank's borrowings or debts.",
                "Tax_Expense": "The amount paid as taxes on profits.",
                "Dividend_Payout": "The portion of earnings distributed to shareholders as dividends."
            }
            self.task = 'call to action'

        elif dataset_name == 'Churn':
            self.train_input = self.train_base
            self.chosen_features = self.train_base.columns
            self.features_description = bank_data_dict = {
                "RowNumber": "Row numbers from 1 to 10000",
                "CustomerId": "Unique IDs for bank customer identification",
                "Surname": "Customer's last name",
                "CreditScore": "Credit score of the customer",
                "Geography": "The country from which the customer belongs",
                "Gender": "Male or Female",
                "Age": "Age of the customer",
                "Tenure": "Number of years for which the customer has been with the bank",
                "Balance": "Bank balance of the customer",
                "NumOfProducts": "Number of bank products the customer is utilising",
                "HasCrCard": "Binary flag for whether the customer holds a credit card with the bank or not",
                "IsActiveMember": "Binary flag for whether the customer is an active member with the bank or not",
                "EstimatedSalary": "Estimated salary of the customer in dollars",
                "Exited": "Binary flag (1 if the customer closed account with the bank, 0 if the customer is retained)"
            }

            self.task = 'classification'



            

    
