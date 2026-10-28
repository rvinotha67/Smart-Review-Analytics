import pandas as pd
from transformers import pipeline
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
from scraper import scrape_reviews
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# -----------------------------
# Sentiment Model
# -----------------------------
classifier = pipeline("sentiment-analysis")


# -----------------------------
# Load Reviews
# -----------------------------
df = pd.read_csv("data/reviews.csv")


# -----------------------------
# Sentiment Analysis
# -----------------------------
sentiments = []

for review in df["review"]:
    result = classifier(review)[0]

    if result["score"] < 0.70:
        sentiments.append("NEUTRAL")
    else:
        sentiments.append(result["label"])


df["Sentiment"] = sentiments


# -----------------------------
# Review Number
# -----------------------------
df["Review Number"] = range(1, len(df) + 1)


# -----------------------------
# Sentiment Score
# -----------------------------
df["Sentiment Score"] = df["Sentiment"].map({
    "NEGATIVE": -1,
    "NEUTRAL": 0,
    "POSITIVE": 1
})


# -----------------------------
# Sentiment Trend
# -----------------------------
trend_fig = px.line(
    df,
    x="Review Number",
    y="Sentiment Score",
    title="Sentiment Trend",
    markers=True
)

trend_fig.update_yaxes(
    tickmode="array",
    tickvals=[-1, 0, 1],
    ticktext=["Negative", "Neutral", "Positive"]
)


# -----------------------------
# Negative Reviews
# -----------------------------
negative_reviews = " ".join(
    df[df["Sentiment"] == "NEGATIVE"]["review"]
)


# -----------------------------
# Negative Keywords
# -----------------------------
words = negative_reviews.lower().split()

stop_words = [
    "is",
    "very",
    "the",
    "a",
    "an",
    "and",
    "of"
]

words = [
    word
    for word in words
    if word not in stop_words
]


common_words = Counter(words).most_common(5)


# -----------------------------
# WordCloud
# -----------------------------
if words:

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(" ".join(words))

    wordcloud.to_file("assets/wordcloud.png")

else:

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate("No negative reviews")


    wordcloud.to_file("assets/wordcloud.png")


# -----------------------------
# Sentiment Counts
# -----------------------------
counts = df["Sentiment"].value_counts()

total_reviews = len(df)

positive_count = counts.get("POSITIVE", 0)
negative_count = counts.get("NEGATIVE", 0)
neutral_count = counts.get("NEUTRAL", 0)


# -----------------------------
# Percentages
# -----------------------------
positive_percent = round(
    (positive_count / total_reviews) * 100,
    2
)

negative_percent = round(
    (negative_count / total_reviews) * 100,
    2
)

neutral_percent = round(
    (neutral_count / total_reviews) * 100,
    2
)


# -----------------------------
# Pie Chart
# -----------------------------
pie_fig = px.pie(
    values=counts.values,
    names=counts.index,
    title="Sentiment Distribution"
)


# -----------------------------
# Bar Chart
# -----------------------------
bar_fig = px.bar(
    x=counts.index,
    y=counts.values,
    title="Sentiment Count"
)


# -----------------------------
# Dash App
# -----------------------------
app = Dash(__name__)


# -----------------------------
# Summary Cards
# -----------------------------
cards = html.Div(
    [

        html.Div(
            [
                html.H3("Total Reviews"),
                html.H2(f"{total_reviews}")
            ],
            style={
                "padding": "20px",
                "border": "1px solid #ddd",
                "borderRadius": "10px",
                "textAlign": "center"
            }
        ),

        html.Div(
            [
                html.H3("Positive"),
                html.H2(f"{positive_count}")
            ],
            style={
                "padding": "20px",
                "border": "1px solid #ddd",
                "borderRadius": "10px",
                "textAlign": "center"
            }
        ),

        html.Div(
            [
                html.H3("Negative"),
                html.H2(f"{negative_count}")
            ],
            style={
                "padding": "20px",
                "border": "1px solid #ddd",
                "borderRadius": "10px",
                "textAlign": "center"
            }
        ),

        html.Div(
            [
                html.H3("Neutral"),
                html.H2(f"{neutral_count}")
            ],
            style={
                "padding": "20px",
                "border": "1px solid #ddd",
                "borderRadius": "10px",
                "textAlign": "center"
            }
        )

    ],

    style={
        "display": "grid",
        "gridTemplateColumns": "repeat(4, 1fr)",
        "gap": "15px",
        "margin": "20px 0"
    }
)


# -----------------------------
# Callback
# -----------------------------
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

    if "amazon." not in product_url and "amzn.in" not in product_url:
        return "Please enter a valid Amazon Product URL."

    reviews = scrape_reviews(
        product_url,
        limit=100
    )

    if not reviews:
        return (
            "Amazon URL is valid, but live reviews could not be fetched. "
            "Amazon is currently blocking automated requests."
        )

    return f"Successfully fetched {len(reviews)} reviews."


# -----------------------------
# Dashboard Layout
# -----------------------------
app.layout = html.Div(

    [

        cards,

        html.H1(
    "Smart Review Analytics Dashboard",
    style={
        "textAlign": "center",
        "marginBottom": "25px"
    }
),
    

        dcc.Input(
    id="product-url",
    type="text",
    placeholder="Enter Amazon Product URL",
    style={
        "width": "60%",
        "padding": "12px",
        "borderRadius": "8px",
        "border": "1px solid #ccc"
    }
),
       html.Button(
    "Analyze Reviews",
    id="analyze-btn",
    style={
        "marginLeft": "10px",
        "padding": "12px 25px",
        "borderRadius": "8px",
        "border": "none",
        "cursor": "pointer",
        "fontSize": "16px"
    }
),

        html.Div(
            id="review-status",
            style={
                "margin": "15px",
                "fontWeight": "bold"
            }
        ),

        dcc.Graph(
            figure=pie_fig
        ),

        dcc.Graph(
            figure=bar_fig
        ),

        dcc.Graph(
            figure=trend_fig
        ),

        html.H2(
            "Top Negative Keywords"
        ),

        html.Ul(
            [
                html.Li(
                    f"{word}: {count}"
                )
                for word, count in common_words
            ]
        ),

        html.H2(
            "Negative WordCloud"
        ),

        html.Img(
            src="/assets/wordcloud.png",
            style={
                "width": "60%",
                "height": "auto"
            }
        ),

        html.H2(
            "Negative Reviews"
        ),

        html.H3(
            f"Total Reviews: {total_reviews}"
        ),

        html.H3(
            f"Positive Reviews: {positive_percent}%"
        ),

        html.H3(
            f"Negative Reviews: {negative_percent}%"
        ),

        html.H3(
            f"Neutral Reviews: {neutral_percent}%"
        ),

        html.Ul(
            [
                html.Li(review)
                for review in
                df[
                    df["Sentiment"] == "NEGATIVE"
                ]["review"]
            ]
        )

    ],

    style={
        "padding": "20px"
    }
)


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(
        debug=True,
        port=8050
    )