# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("tested.csv")

def data_preprocessing(data):
# Data Exploration
    print(data,data.shape,data.info(),data.isnull().sum())
    """Preprocess the data by removing 'Cabin', filling missing values, and converting categorical values."""
    # Remove the 'Cabin' column
    data.drop(columns='Cabin', inplace=True)

    # Fill missing values
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data['Fare'].fillna(data['Fare'].mode()[0], inplace=True)

    # Transform categorical values to numerical values
    data.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)

def prepare_data(data):
    """Prepare data for modeling by separating features (X) and the target (Y)."""
    X = data.drop(columns=['PassengerId', 'Name', 'Ticket'], axis=1)
    Y = data['Survived']
    return X, Y

def train_and_evaluate_model(X_train, X_test, Y_train, Y_test):
    """Train a logistic regression model and evaluate its accuracy on training and testing data."""
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    Y_train_predictions = model.predict(X_train)
    training_accuracy = accuracy_score(Y_train, Y_train_predictions)
    print("Accuracy on training data:", training_accuracy)

    Y_test_predictions = model.predict(X_test)
    testing_accuracy = accuracy_score(Y_test, Y_test_predictions)
    print("Accuracy on testing data:", testing_accuracy)

# Data Visualization
def set_custom_palette(palette_name):
    # Define custom color palettes
    custom_palettes = {
        "Age Distribution": "Set1",
        "Pairplot": "viridis",
        "Correlation Heatmap": "coolwarm",
        "Survival Rate by Passenger Class": "pastel",
        "Survival Rate by Gender": "husl",
        "Survival by Port of Embarkation": "muted"
    }
    
    # Set the custom palette for the specific plot
    sns.set_palette(custom_palettes[palette_name])

def plot_count_plots(data):
    # Set the style for the plots
    sns.set()

    # Create subplots for count plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Count Plot for 'Survived'
    sns.countplot(x='Survived', data=data, ax=axes[0, 0])
    axes[0, 0].set_title('Survival Count')

    # Count Plot for 'Sex'
    sns.countplot(x='Sex', data=data, ax=axes[0, 1])
    axes[0, 1].set_title('Gender Count')

    # Count Plot for 'Sex' with 'Survived' hue
    sns.countplot(x='Sex', hue='Survived', data=data, ax=axes[1, 0])
    axes[1, 0].set_title('Survival by Gender')

    # Count Plot for 'Pclass'
    sns.countplot(x='Pclass', data=data, ax=axes[1, 1])
    axes[1, 1].set_title('Passenger Class Count')

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()

def plot_age_distribution(data):
    set_custom_palette("Age Distribution")
    plt.figure(figsize=(8, 6))
    sns.histplot(data=data, x='Age', bins=30, kde=True)
    plt.title('Distribution of Ages')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.show()

def plot_pairplot(data):
    set_custom_palette("Pairplot")
    sns.pairplot(data=data, hue='Survived', diag_kind='kde')
    plt.show()

def plot_correlation_heatmap(data):
    set_custom_palette("Correlation Heatmap")
    data_for_correlation = data[['Age', 'SibSp', 'Parch', 'Fare']]
    correlation_matrix = data_for_correlation.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title('Correlation Heatmap')
    plt.show()

def plot_survival_rate_by_pclass(data):
    set_custom_palette("Survival Rate by Passenger Class")
    sns.barplot(x='Pclass', y='Survived', data=data)
    plt.title('Survival Rate by Passenger Class')
    plt.show()

def plot_survival_rate_by_gender(data):
    set_custom_palette("Survival Rate by Gender")
    sns.barplot(x='Sex', y='Survived', data=data)
    plt.title('Survival Rate by Gender')
    plt.show()

def plot_survival_by_port_of_embarkation(data):
    set_custom_palette("Survival by Port of Embarkation")
    sns.countplot(x='Embarked', hue='Survived', data=data)
    plt.title('Survival by Port of Embarkation')
    plt.show()

data_preprocessing(data)
X, Y = prepare_data(data)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
train_and_evaluate_model(X_train, X_test, Y_train, Y_test)

plot_count_plots(data)
plot_age_distribution(data)
plot_pairplot(data)
plot_correlation_heatmap(data)
plot_survival_rate_by_pclass(data)
plot_survival_rate_by_gender(data)
plot_survival_by_port_of_embarkation(data)