import pandas as pd
from transformers import pipeline
import plotly.express as px

classifier = pipeline("sentiment-analysis")

df = pd.read_csv("data/reviews.csv")

sentiments = []

for review in df["review"]:
    result = classifier(review)
    label = result[0]["label"]

    if result[0]["score"] < 0.90:
        sentiments.append("NEUTRAL")
    else:
        sentiments.append(label)

df["Sentiment"] = sentiments
print(df)

counts = df["Sentiment"].value_counts()

fig = px.pie(
    values=counts.values,
    names=counts.index,
    title="Sentiment Distribution"
)

fig.show()