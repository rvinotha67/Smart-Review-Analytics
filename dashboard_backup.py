import pandas as pd
from transformers import pipeline
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
from scraper import scrape_reviews
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

classifier = pipeline("sentiment-analysis")

df = pd.read_csv("data/reviews.csv")

sentiments = []

for review in df["review"]:
    result = classifier(review)
    sentiments.append(result[0]["label"])

df["Sentiment"] = sentiments

negative_reviews = " ".join(
    df[df["Sentiment"] == "NEGATIVE"]["review"]
)

words = negative_reviews.lower().split()
stop_words = ["is", "very", "the", "a", "an", "and", "of"]

words = [
    word for word in words
    if word not in stop_words
]

common_words = Counter(words).most_common(5)
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(" ".join(words))
wordcloud.to_file("assets/wordcloud.png")

plt.imshow(wordcloud)
plt.axis("off")
plt.show()


counts = df["Sentiment"].value_counts()
total_reviews = len(df)

positive_percent = round(
    (counts.get("POSITIVE", 0) / total_reviews) * 100,
    2
)

negative_percent = round(
    (counts.get("NEGATIVE", 0) / total_reviews) * 100,
    2
)

pie_fig = px.pie(
    values=counts.values,
    names=counts.index,
    title="Sentiment Distribution"
)

bar_fig = px.bar(
    x=counts.index,
    y=counts.values,
    title="Sentiment Count"
)

app = Dash(__name__)
@app.callback(
    Output("review-status", "children"),
    Input("analyze-btn", "n_clicks"),
    Input("product-url", "value")
)
def analyze_reviews(n_clicks, product_url):

    if not n_clicks:
        return "Enter an Amazon Product URL and click Analyze Reviews."

    if not product_url:
        return "Please enter an Amazon Product URL."

    reviews = scrape_reviews(product_url, limit=100)

    if not reviews:
        return "No reviews found. Please check the Amazon Product URL."

    return f"Successfully fetched {len(reviews)} reviews."

app.layout = html.Div([
    html.H1("Smart Review Analytics Dashboard"),

    dcc.Input(
    id="product-url",
    type="text",
    placeholder="Enter Product URL",
    style={"width": "60%"}
),

html.Button("Analyze Reviews", id="analyze-btn"),
html.Div(
    id="review-status",
    style={"margin": "15px", "fontWeight": "bold"}
),

    dcc.Graph(figure=pie_fig),

    dcc.Graph(figure=bar_fig),

    html.H2("Top Negative Keywords"),
    
html.Ul([
    html.Li(f"{word}: {count}")
    for word, count in common_words
   ]),
       html.H2("Negative WordCloud"),
    html.Img(src="/assets/wordcloud.png", style={"width": "60%", "height": "auto"}),
   html.H2("Negative Reviews"),
   html.H3(f"Total Reviews: {total_reviews}"),

html.H3(f"Positive Reviews: {positive_percent}%"),

html.H3(f"Negative Reviews: {negative_percent}%"),

html.Ul([
    html.Li(review)
    for review in df[df["Sentiment"] == "NEGATIVE"]["review"]
])
])

if __name__ == "__main__":
    app.run(debug=True)
    