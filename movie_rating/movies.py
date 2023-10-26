import pandas as pd
import matplotlib.pyplot as mpl
import seaborn as sb
import plotly.express as px
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.linear_model import LinearRegression

movie_file=pd.read_csv("IMDb.csv", encoding='latin1')

def display_basic_stats(data):
    print(data.head(11))
    print(data.describe())
    print(data.dtypes)
    print(data.isnull().sum())
    print(data.isnull().sum().sum())
    print(data.shape)

def preprocess_data(data):
    data.dropna(inplace=True)
    data['Year'] = data['Year'].str.extract('(\d+)').astype(int)
    data['Votes'] = data['Votes'].str.replace(',', '').astype(int)

def analyze_genre(data):
    genres = data['Genre'].str.split(', ', expand=True)
    genre_counts = genres.stack().value_counts()
    wordcloud = WordCloud(width=950, height=550, background_color='white').generate_from_frequencies(genre_counts)
    mpl.figure(figsize=(16, 6))
    mpl.imshow(wordcloud, interpolation='bilinear')
    mpl.axis('off')
    mpl.title('Genre Word Cloud')
     
def piechart():
    genre = movie_file['Genre']
    genre.head(11)
    genres = movie_file['Genre'].str.split(', ', expand=True)
    genres.head(11)
    genre_counts = {}
    for genre in genres.values.flatten():
        if genre is not None:
            if genre in genre_counts:
                genre_counts[genre] += 1
            else:
                genre_counts[genre] = 1

    genereCounts = {genre: count for genre, count in sorted(genre_counts.items())}

    for genre, count in genereCounts.items():
        print(f"{genre}: {count}")

    genresPie = movie_file['Genre'].value_counts()
    genresPie.head(11)

    genrePie = pd.DataFrame(list(genresPie.items()))
    genrePie = genrePie.rename(columns={0: 'Genre', 1: 'Count'})
    genrePie.head(11)
    genrePie.loc[genrePie['Count'] < 50, 'Genre'] = 'Other'
    ax = px.pie(genrePie, values='Count', names='Genre', title='More than one Genre of movies in Indian Cinema')
    ax.show()

def analyze_ratings(data):
    data['Rating']=data['Rating'].astype('category').cat.codes
    Q1 = data['Rating'].quantile(0.25)
    Q3 = data['Rating'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    filtered_data = data[(data['Rating'] >= lower_bound) & (data['Rating'] <= upper_bound)]

    sb.histplot(data=filtered_data, x="Rating", bins=20, kde=True)
    mpl.show()


def analyze_directors(data):
    directors = data["Director"].value_counts().head(20)
    sb.barplot(x=directors.index, y=directors.values, palette='viridis')
    mpl.xlabel('Directors')
    mpl.ylabel('Frequency of Movies')
    mpl.title('Top 20 Directors by Frequency of Movies')
    mpl.xticks(rotation=90)
    mpl.show()

def analyze_actors(data):
    actors = pd.concat([data['Actor 1'], data['Actor 2'], data['Actor 3']]).dropna().value_counts().head(20)
    sb.barplot(x=actors.index, y=actors.values, palette='viridis')
    mpl.xlabel('Actors')
    mpl.ylabel('Total Number of Movies')
    mpl.title('Top 20 Actors with Total Number of Movies')
    mpl.xticks(rotation=90)
    mpl.show()

def analyze_duration(data):
    data['Duration'] = data['Duration'].str.extract('(\d+)').astype(float)
    data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
    sb.lineplot(data=data.groupby('Year')['Duration'].mean().reset_index(), x='Year', y='Duration')
    mpl.show()


def filter_outliers(data, column):
    if column=="Actor":
        data["Actor"] = data['Actor 1'] + ', ' + data['Actor 2'] + ', ' + data['Actor 3']
    
    data[column]=data[column].astype('category').cat.codes  

    ax = sb.boxplot(data=data, y=column)
    ax.set_ylabel(column)
    ax.set_title(f'Box Plot of {column}')
    mpl.show()

    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]


def prepare_data_and_evaluate(data, model, model_name):
    # Prepare the data
    x = data.drop(['Name', 'Genre', 'Rating', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Actor'], axis=1)
    y = data['Rating']

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # Train the model
    model.fit(x_train, y_train)

    # Make predictions
    y_pred = model.predict(x_test)

    # Evaluate model performance
    acc = round(r2_score(y_test, y_pred) * 100, 2)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("Model:", model_name)
    print("Accuracy = {:0.2f}%".format(acc))
    print("Root Mean Squared Error = {:0.2f}\n".format(rmse))


if __name__ == '__main__':
    display_basic_stats(movie_file)
    preprocess_data(movie_file)
    analyze_genre(movie_file)
    piechart()
    analyze_ratings(movie_file)
    analyze_directors(movie_file)
    analyze_actors(movie_file)
    analyze_duration(movie_file)

    movie_file = filter_outliers(movie_file, 'Genre')
    movie_file = filter_outliers(movie_file, 'Director')
    movie_file = filter_outliers(movie_file, 'Actor')

    model = LinearRegression()
    prepare_data_and_evaluate(movie_file, model, "Linear Regression")