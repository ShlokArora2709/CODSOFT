import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import re
from sklearn.preprocessing import MinMaxScaler


data=pd.read_csv("car_purchasing1.csv")

    
def preprocessing(data):
    data['age'].fillna(data['age'].mean(), inplace=True)
    data['annual Salary'].fillna(data['annual Salary'].mean(),inplace=True)
    data['credit card debt'].fillna(data['credit card debt'].mean(),inplace=True)
    data['net worth'].fillna(data['net worth'].mean(),inplace=True)
    data['car purchase amount'].fillna(data['car purchase amount'].mean(),inplace=True)

def gender_bar(data):
    gender_counts = data['gender'].value_counts()
    gender_counts.plot(kind='bar', rot=0, color=['skyblue', 'lightcoral'])
    plt.title('Gender Distribution')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.show()

def age_hist(data):
    data['age'].plot(kind='hist', bins=20, color='orange', edgecolor='black')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()

def scatter_income(data):
    plt.scatter(data['annual Salary'], data['net worth'], color='green')
    plt.title('Net Worth vs. Annual Salary')
    plt.xlabel('Annual Salary')
    plt.ylabel('Net Worth')
    plt.show()

def car_by_country(data):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='country', y='car purchase amount', data=data, palette='viridis')
    plt.title('Car Purchase Amount by Country')
    plt.xlabel('Country')
    plt.ylabel('Car Purchase Amount')
    plt.xticks(rotation=45)
    plt.show()

def gender_pie(data):
    lable={0:"Female",1:"Male"}
    gender_proportion = data['gender'].map(lable).value_counts()
    plt.pie(gender_proportion, labels=gender_proportion.index, autopct='%1.1f%%', colors=['skyblue','lightcoral'])
    plt.title('Gender Proportion')
    plt.show()

def extract_code_from_emails(df):

    def extract_code(email):
        match = re.search(r'\.([a-zA-Z]{2,3})$', email)
        return match.group(1) if match else None

    # Apply the function to the specified column and create a new column 'country_code'
    df['email_code'] = df['customer e-mail'].apply(extract_code)

    return df

def add_columns_for_info(data):
    data['fiancial_stability_ratio']=data['annual Salary']/(data['credit card debt']+data['net worth'])
    data['networth_age_ratio']=data['net worth']/data['age']

    salary_weight = 0.6
    debt_weight = 0.2
    net_worth_weight = 0.2

# Calculate Financial Stability Index
    data['financial_stability_index'] = (salary_weight * data['annual Salary'] +
    debt_weight * data['credit card debt'] -
    net_worth_weight * data['net worth'])
    scaler = MinMaxScaler()
    data[['networth_age_ratio','fiancial_stability_ratio','financial_stability_index']] = scaler.fit_transform(data[['networth_age_ratio', 'fiancial_stability_ratio', 'financial_stability_index']])

def scatter_age(data):
    plt.scatter(data['age'], data['net worth'], color='blue',s=50,marker="X")
    plt.title('Net Worth vs. Age')
    plt.xlabel('Age')
    plt.ylabel('Net Worth')
    plt.show()

def prepare_data(data):
    # One-hot encode categorical variables
    data_encoded = pd.get_dummies(data, columns=['country', 'gender'], drop_first=True)
    
    X = data_encoded[['age', 'annual Salary', 'credit card debt', 'net worth']]
    Y = data_encoded['car purchase amount']
    return X, Y

def train_and_evaluate_model(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    """Train a linear regression model and evaluate its accuracy on training and testing data."""
    model = LinearRegression()
    model.fit(X_train, Y_train)

    Y_train_predictions = model.predict(X_train)
    training_mse = mean_squared_error(Y_train, Y_train_predictions)
    print("Mean Squared Error on training data:", training_mse)

    Y_test_predictions = model.predict(X_test)
    testing_mse = mean_squared_error(Y_test, Y_test_predictions)
    print("Mean Squared Error on testing data:", testing_mse)


preprocessing(data)
scatter_age(data)
gender_bar(data)
gender_pie(data)
age_hist(data)
scatter_age(data)
scatter_income(data)
car_by_country(data)
add_columns_for_info(data)
df=extract_code_from_emails(data)
print(df)
X,Y=prepare_data(data)
train_and_evaluate_model(X,Y)