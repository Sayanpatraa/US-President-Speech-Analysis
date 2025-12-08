import os 

print(os.getcwd())

os.chdir(r"C:\Users\shime\Downloads")
#%%
import json
import pandas as pd

# Path to the folder containing all the JSON speech files
folder_path = r"C:\Users\shime\Downloads\speeches" #Change path as needed 

# Read and store all speeches
speeches = []
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
            data = json.load(f)
            speeches.append({
                "title": data.get("title"),
                "date": data.get("date"),
                "url": data.get("url"),
                "transcript": data.get("transcript"),
                "president": data.get("president")
            })

# Convert to DataFrame
df = pd.DataFrame(speeches)
print(df.shape)
print(df.head())

# %%
print(df.tail())


# Convert date column to datetime (handles timezone offsets)
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Check the most recent date
latest_date = df['date'].max()
earliest_date = df['date'].min()

print("Earliest speech date:", earliest_date)
print("Most recent speech date:", latest_date)

# See which speech that corresponds to
latest_speech = df.loc[df['date'] == latest_date]
print("\nLatest speech info:\n", latest_speech[['title', 'date', 'url']])

#%%

#Make copy of df to keep original intact
df_raw = df.copy()

'''From this point on we will manipulate df and keep df_raw as original'''


#Check df again
print(df.columns)

#Check presidents in dataset
print(df['president'].unique())
print(len(df['president'].unique()))


#Add party affiliation
party_affiliation = { "Donald Trump": "Republican",
                        "Joe Biden": "Democrat",
                        "Barack Obama": "Democrat",
                        "George W. Bush": "Republican",
                        "Bill Clinton": "Democrat",
                        "George H. W. Bush": "Republican",
                        "Ronald Reagan": "Republican",
                        "Jimmy Carter": "Democrat",
                        "Gerald Ford": "Republican",
                        "Richard M. Nixon": "Republican",
                        "Lyndon B. Johnson": "Democrat",
                        "John F. Kennedy": "Democrat",
                        "Dwight D. Eisenhower": "Republican",
                        "Harry S. Truman": "Democrat",
                        "Franklin D. Roosevelt": "Democrat",
                        "Herbert Hoover": "Republican", 
                        "Calvin Coolidge": "Republican",
                        "Warren G. Harding": "Republican",
                        "Woodrow Wilson": "Democrat",
                        "William Taft": "Republican",
                        "Theodore Roosevelt": "Republican",
                        "William McKinley": "Republican",
                        "Grover Cleveland": "Democrat",
                        "Benjamin Harrison": "Republican",
                        "Chester A. Arthur": "Republican",
                        "James A. Garfield": "Republican",
                        "Rutherford B. Hayes": "Republican",
                        "Ulysses S. Grant": "Republican",
                        "Andrew Johnson": "Democrat",
                        "Abraham Lincoln": "Republican",
                        "James Buchanan": "Democrat",
                        "Franklin Pierce": "Democrat",
                        "Millard Fillmore": "Whig",
                        "Zachary Taylor": "Whig",
                        "James K. Polk": "Democrat",
                        "John Tyler": "Whig",
                        "William Harrison": "Whig",
                        "Martin Van Buren": "Democrat",
                        "Andrew Jackson": "Democrat",
                        "John Quincy Adams": "National Republican",
                        "James Monroe": "Democrat-Republican",
                        "James Madison": "Democrat-Republican",
                        "Thomas Jefferson": "Democrat-Republican",
                        "John Adams": "Federalist",
                        "George Washington": "Federalist"
                        }

#Check if we have all presidents listed
print(len(party_affiliation))

#Map party affiliation to df
df['party'] = df['president'].map(party_affiliation)

#Check if it correctly mapped
print(df[['president', 'party']].drop_duplicates().sort_values(by='president'))


'''Prior to the two party system we know, there were other parties such as Whig, Federalist, National Republican, and Democrat-Republican. We will keep these as is for now.'''

#%%

#Check counts of speeches by president
print(df['president'].value_counts())

'''We can see that some presidents have very few speeches, especially those prior to the 20th century. This may affect our analysis later on, so we should keep this in mind.'''

#%%

#Add term dates column
def term_limit(president):
    term_limits = {
        "Donald Trump": ("2017-01-20", "2021-01-20"),
        "Joe Biden": ("2021-01-20", "2025-01-20"),
        "Barack Obama": ("2009-01-20", "2017-01-20"),
        "George W. Bush": ("2001-01-20", "2009-01-20"),
        "Bill Clinton": ("1993-01-20", "2001-01-20"),
        "George H. W. Bush": ("1989-01-20", "1993-01-20"),
        "Ronald Reagan": ("1981-01-20", "1989-01-20"),
        "Jimmy Carter": ("1977-01-20", "1981-01-20"),
        "Gerald Ford": ("1974-08-09", "1977-01-20"),
        "Richard M. Nixon": ("1969-01-20", "1974-08-09"),
        "Lyndon B. Johnson": ("1963-11-22", "1969-01-20"),
        "John F. Kennedy": ("1961-01-20", "1963-11-22"),
        "Dwight D. Eisenhower": ("1953-01-20", "1961-01-20"),
        "Harry S. Truman": ("1945-04-12", "1953-01-20"),
        "Franklin D. Roosevelt": ("1933-03-04", "1945-04-12"),
        "Herbert Hoover": ("1929-03-04", "1933-03-04"),
        "Calvin Coolidge": ("1923-08-02", "1929-03-04"),
        "Warren G. Harding": ("1921-03-04", "1923-08-02"),
        "Woodrow Wilson": ("1913-03-04", "1921-03-04"),
        "William Taft": ("1909-03-04", "1913-03-04"),
        "Theodore Roosevelt": ("1901-09-14", "1909-03-04"),
        "William McKinley": ("1897-03-04", "1901-09-14"),
        "Grover Cleveland": ("1893-03-04", "1897-03-04"),
        "Benjamin Harrison": ("1889-03-04", "1893-03-04"),
        "Chester A. Arthur": ("1881-09-19", "1885-03-04"),
        "James A. Garfield": ("1881-03-04", "1881-09-19"),
        "Rutherford B. Hayes": ("1877-03-04", "1881-03-04"),
        "Ulysses S. Grant": ("1869-03-04", "1877-03-04"),
        "Andrew Johnson": ("1865-04-15", "1869-03-04"),
        "Abraham Lincoln": ("1861-03-04", "1865-04-15"),
        "James Buchanan": ("1857-03-04", "1861-03-04"),
        "Franklin Pierce": ("1853-03-04", "1857-03-04"),
        "Millard Fillmore": ("1850-07-09", "1853-03-04"),
        "Zachary Taylor": ("1849-03-04", "1850-07-09"),
        "James K. Polk": ("1845-03-04", "1849-03-04"),
        "John Tyler": ("1841-04-04", "1845-03-04"),
        "William Harrison": ("1841-03-04", "1841-04-04"),
        "Martin Van Buren": ("1837-03-04", "1841-03-04"),
        "Andrew Jackson": ("1829-03-04", "1837-03-04"),
        "John Quincy Adams": ("1825-03-04", "1829-03-04"),
        "James Monroe": ("1817-03-04", "1825-03-04"),
        "James Madison": ("1809-03-04", "1817-03-04"),
        "Thomas Jefferson": ("1801-03-04", "1809-03-04"),
        "John Adams": ("1797-03-04", "1801-03-04"),
        "George Washington": ("1789-04-30", "1797-03-04")
    }
    return term_limits.get(president, (None, None))

#Apply terms to df
df['term_start'], df['term_end'] = zip(*df['president'].apply(term_limit))

#Check if it correctly mapped
print(df[['president', 'term_start', 'term_end']].drop_duplicates().sort_values(by='president'))

#Look at df again
print(df.head())
#%%
#Preprocess text

#Import necessary libraries
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords') # Download if needed
from nltk.tokenize import word_tokenize
nltk.download('punkt') #Download if needed
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet') #Download if needed

#Define stopwords
stopwords = set(stopwords.words('english'))

#Create function to preprocess text
def preprocess(text): 
    text = text.lower()  # Lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'&[a-z]+;', ' ', text)  # Remove HTML entities
    text = re.sub(r"[^a-z\s']", ' ', text)  # Remove punctuation and special characters
    text = re.sub(r'\s+', ' ',text).strip()  # Remove extra whitespace
    text = re.sub (r'\d+', '', text)  # Remove numbers
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in stopwords]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return ' '.join(tokens)  # Join tokens back to string

#Apply to text 
df['cleaned_transcript'] = df['transcript'].apply(preprocess)

#Check df 
print(df[['transcript', 'cleaned_transcript']].head())
#%%# 
# Topic analysis will be done in next steps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation

#Vectorization and fitting
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(df['cleaned_transcript'])

# Fit NMF model
num_topics = 10  # Define number of topics
nmf_model = NMF(n_components=num_topics, random_state=42)
nmf_model.fit(X)

# Display topics
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx + 1}:")
        print(" | ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        print()

#Use the function to display topics
no_top_words = 10
tfidf_feature_names = vectorizer.get_feature_names_out()
display_topics(nmf_model, tfidf_feature_names, no_top_words)