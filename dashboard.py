import os
import re
import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, Input, Output
from transformers import pipeline
from wordcloud import WordCloud


# ============================================================
# SMART REVIEW ANALYTICS DASHBOARD
# Uses exactly 100 unique reviews from:
# data/reviews_100.csv
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "reviews_100.csv"
)

ASSET_DIR = os.path.join(
    BASE_DIR,
    "assets"
)

WORDCLOUD_FILE = os.path.join(
    ASSET_DIR,
    "negative_wordcloud.png"
)

os.makedirs(ASSET_DIR, exist_ok=True)


# ============================================================
# DASH APP
# ============================================================

app = Dash(__name__)

app.title = "Smart Review Analytics"


# ============================================================
# FINBERT MODEL
# ============================================================

MODEL = None


def get_model():

    global MODEL

    if MODEL is None:

        MODEL = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert"
        )

    return MODEL


# ============================================================
# FIND REVIEW COLUMN
# ============================================================

def find_review_column(df):

    names = [
        "Review",
        "review",
        "Review Text",
        "review_text",
        "text",
        "Text",
        "content",
        "Content"
    ]

    for name in names:

        if name in df.columns:

            return name


    # Fallback
    cols = df.select_dtypes(
        include=["object"]
    ).columns.tolist()


    if not cols:

        return None


    return max(
        cols,
        key=lambda c:
        df[c]
        .fillna("")
        .astype(str)
        .str.len()
        .mean()
    )


# ============================================================
# LOAD EXACTLY 100 REVIEWS
# ============================================================

def load_100_reviews():

    if not os.path.exists(DATA_FILE):

        return pd.DataFrame({
            "Review": []
        })


    df = pd.read_csv(DATA_FILE)

    col = find_review_column(df)


    if col is None:

        return pd.DataFrame({
            "Review": []
        })


    df = df.copy()


    df["Review"] = (
        df[col]
        .fillna("")
        .astype(str)
        .str.strip()
    )


    # Remove empty reviews
    df = df[
        df["Review"].str.len() > 0
    ]


    # Remove duplicate reviews
    df = df.drop_duplicates(
        subset=["Review"]
    )


    # EXACTLY FIRST 100 UNIQUE REVIEWS
    df = df.head(100)


    # Review number
    df["Review Number"] = range(
        1,
        len(df) + 1
    )


    return df


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

def analyze(df):

    df = df.copy()


    if df.empty:

        return df


    model = get_model()


    texts = df["Review"].tolist()

    results = []


    # Process 16 reviews at a time
    for i in range(
        0,
        len(texts),
        16
    ):

        batch = texts[
            i:i + 16
        ]


        results.extend(
            model(
                batch,
                truncation=True,
                max_length=512
            )
        )


    # Sentiment label
    df["Sentiment"] = [
        r["label"].upper()
        for r in results
    ]


    # Sentiment score
    df["Sentiment Score"] = [
        round(
            float(r["score"]) * 100,
            2
        )
        for r in results
    ]


    return df


# ============================================================
# WORDCLOUD
# ============================================================

def make_wordcloud(df):

    negative = df.loc[
        df["Sentiment"].eq("NEGATIVE"),
        "Review"
    ].astype(str).tolist()


    text = " ".join(
        negative
    )


    # If no negative reviews
    if not text.strip():

        text = "No negative reviews"


    stopwords = {

        "the",
        "a",
        "an",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "this",
        "that",
        "it",
        "i",
        "my",
        "me",
        "very",
        "really",
        "have",
        "has",
        "had",
        "but",
        "as",
        "be",
        "been",
        "they",
        "them",
        "their",
        "you",
        "your",
        "we",
        "our",
        "from",
        "at",
        "so"
    }


    wc = WordCloud(

        width=1000,

        height=500,

        background_color="white",

        stopwords=stopwords,

        min_font_size=12
    ).generate(text)


    wc.to_file(
        WORDCLOUD_FILE
    )


# ============================================================
# EMPTY FIGURE
# ============================================================

def empty_fig(title):

    fig = px.bar(
        title=title
    )

    fig.update_layout(
        template="plotly_white"
    )

    return fig


# ============================================================
# DASHBOARD LAYOUT
# ============================================================

app.layout = html.Div(

    [

        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        html.H1(

            "Smart Review Analytics Dashboard",

            style={
                "textAlign": "center"
            }
        ),


        # ----------------------------------------------------
        # URL INPUT + BUTTON
        # ----------------------------------------------------

        html.Div(

            [

                dcc.Input(

                    id="product-url",

                    type="url",

                    value="https://web-scraping.dev/reviews",

                    style={

                        "width": "65%",

                        "padding": "12px",

                        "fontSize": "16px"
                    }
                ),


                html.Button(

                    "Analyze Reviews",

                    id="analyze-button",

                    n_clicks=0,

                    style={

                        "padding": "12px 22px",

                        "marginLeft": "10px",

                        "fontSize": "16px"
                    }
                )

            ],

            style={

                "textAlign": "center",

                "marginBottom": "15px"
            }
        ),


        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        html.Div(

            id="status-message",

            style={

                "textAlign": "center",

                "fontWeight": "bold",

                "marginBottom": "20px"
            }
        ),


        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        dcc.Store(
            id="reviews-store"
        ),


        # ----------------------------------------------------
        # SUMMARY CARDS
        # ----------------------------------------------------

        html.Div(

            [

                html.Div(

                    [

                        html.H3(
                            "Total Reviews"
                        ),

                        html.H2(
                            id="total-reviews"
                        )

                    ],

                    className="card"
                ),


                html.Div(

                    [

                        html.H3(
                            "Positive"
                        ),

                        html.H2(
                            id="positive-reviews"
                        )

                    ],

                    className="card"
                ),


                html.Div(

                    [

                        html.H3(
                            "Negative"
                        ),

                        html.H2(
                            id="negative-reviews"
                        )

                    ],

                    className="card"
                ),


                html.Div(

                    [

                        html.H3(
                            "Neutral"
                        ),

                        html.H2(
                            id="neutral-reviews"
                        )

                    ],

                    className="card"
                )

            ],

            style={

                "display": "grid",

                "gridTemplateColumns":
                    "repeat(4, 1fr)",

                "gap": "15px",

                "marginBottom": "30px"
            }
        ),


        # ----------------------------------------------------
        # SENTIMENT DISTRIBUTION
        # ----------------------------------------------------

        html.H2(
            "Sentiment Distribution"
        ),

        dcc.Graph(
            id="sentiment-pie"
        ),


        # ----------------------------------------------------
        # SENTIMENT TREND
        # ----------------------------------------------------

        html.H2(
            "Sentiment Trend"
        ),

        dcc.Graph(
            id="sentiment-trend"
        ),


        # ----------------------------------------------------
        # SENTIMENT COUNT
        # ----------------------------------------------------

        html.H2(
            "Sentiment Count"
        ),

        dcc.Graph(
            id="sentiment-count"
        ),


        # ----------------------------------------------------
        # NEGATIVE KEYWORDS
        # ----------------------------------------------------

        html.H2(
            "Top Negative Keywords"
        ),

        html.Ul(
            id="negative-keywords"
        ),


        # ----------------------------------------------------
        # WORDCLOUD
        # ----------------------------------------------------

        html.H2(
            "Negative WordCloud"
        ),

        html.Img(

            id="wordcloud-image",

            style={

                "width": "100%",

                "maxWidth": "1000px",

                "display": "block",

                "margin": "auto"
            }
        ),


        # ----------------------------------------------------
        # NEGATIVE REVIEWS
        # ----------------------------------------------------

        html.H2(
            "Negative Reviews"
        ),

        html.Div(
            id="negative-review-list"
        )

    ],

    style={

        "maxWidth": "1100px",

        "margin": "auto",

        "padding": "30px",

        "fontFamily":
            "Arial, sans-serif"
    }
)


# ============================================================
# ANALYZE BUTTON CALLBACK
# ============================================================

@app.callback(

    Output(
        "reviews-store",
        "data"
    ),

    Output(
        "status-message",
        "children"
    ),

    Input(
        "analyze-button",
        "n_clicks"
    ),

    prevent_initial_call=True
)

def run_analysis(n_clicks):

    try:

        # Load prepared 100 reviews
        df = load_100_reviews()


        if len(df) == 0:

            return (

                None,

                "ERROR: No reviews found in data/reviews_100.csv"
            )


        # FinBERT analysis
        df = analyze(df)


        # Save analyzed data
        output_file = os.path.join(

            BASE_DIR,

            "data",

            "reviews_100_analyzed.csv"
        )


        df.to_csv(

            output_file,

            index=False
        )


        # Create wordcloud
        make_wordcloud(df)


        # Counts
        positive = int(
            (
                df["Sentiment"]
                == "POSITIVE"
            ).sum()
        )


        negative = int(
            (
                df["Sentiment"]
                == "NEGATIVE"
            ).sum()
        )


        neutral = int(
            (
                df["Sentiment"]
                == "NEUTRAL"
            ).sum()
        )


        message = (

            f"Successfully loaded "
            f"{len(df)} reviews and completed "
            f"FinBERT sentiment analysis. "

            f"Positive: {positive}, "

            f"Negative: {negative}, "

            f"Neutral: {neutral}"
        )


        return (

            df.to_dict(
                "records"
            ),

            message
        )


    except Exception as e:

        return (

            None,

            f"Error: {type(e).__name__}: {e}"
        )


# ============================================================
# DASHBOARD OUTPUT CALLBACK
# ============================================================

@app.callback(

    Output(
        "total-reviews",
        "children"
    ),

    Output(
        "positive-reviews",
        "children"
    ),

    Output(
        "negative-reviews",
        "children"
    ),

    Output(
        "neutral-reviews",
        "children"
    ),

    Output(
        "sentiment-pie",
        "figure"
    ),

    Output(
        "sentiment-trend",
        "figure"
    ),

    Output(
        "sentiment-count",
        "figure"
    ),

    Output(
        "negative-keywords",
        "children"
    ),

    Output(
        "negative-review-list",
        "children"
    ),

    Output(
        "wordcloud-image",
        "src"
    ),

    Input(
        "reviews-store",
        "data"
    )
)

def update_dashboard(data):

    # --------------------------------------------------------
    # NO DATA
    # --------------------------------------------------------

    if not data:

        empty = empty_fig(
            "Click Analyze Reviews"
        )


        return (

            0,

            0,

            0,

            0,

            empty,

            empty,

            empty,

            [],

            [],

            ""
        )


    df = pd.DataFrame(
        data
    )


    # --------------------------------------------------------
    # SENTIMENT COUNTS
    # --------------------------------------------------------

    counts = (

        df["Sentiment"]

        .value_counts()

        .reindex(
            [
                "POSITIVE",
                "NEGATIVE",
                "NEUTRAL"
            ]
        )

        .fillna(0)

        .astype(int)
    )


    chart_df = pd.DataFrame({

        "Sentiment":
            counts.index,

        "Count":
            counts.values
    })


    # --------------------------------------------------------
    # PIE CHART
    # --------------------------------------------------------

    pie = px.pie(

        chart_df,

        names="Sentiment",

        values="Count",

        title=
            "Sentiment Distribution"
    )


    # --------------------------------------------------------
    # SENTIMENT TREND
    # --------------------------------------------------------

    values = {

        "NEGATIVE": -1,

        "NEUTRAL": 0,

        "POSITIVE": 1
    }


    trend = df.copy()


    trend["Sentiment Value"] = (

        trend["Sentiment"]
        .map(values)
    )


    trend_fig = px.line(

        trend,

        x="Review Number",

        y="Sentiment Value",

        markers=True,

        title=
            "Sentiment Trend"
    )


    trend_fig.update_yaxes(

        tickmode="array",

        tickvals=[
            -1,
            0,
            1
        ],

        ticktext=[
            "Negative",
            "Neutral",
            "Positive"
        ],

        title="Sentiment"
    )


    # --------------------------------------------------------
    # COUNT CHART
    # --------------------------------------------------------

    count_fig = px.bar(

        chart_df,

        x="Sentiment",

        y="Count",

        title=
            "Sentiment Count"
    )


    # --------------------------------------------------------
    # NEGATIVE KEYWORDS
    # --------------------------------------------------------

    negative_text = " ".join(

        df.loc[
            df["Sentiment"]
            == "NEGATIVE",
            "Review"
        ]

        .astype(str)

        .tolist()
    ).lower()


    words = re.findall(

        r"\b[a-zA-Z]{4,}\b",

        negative_text
    )


    stopwords = {

        "this",
        "that",
        "with",
        "from",
        "have",
        "very",
        "were",
        "they",
        "their",
        "there",
        "about",
        "would",
        "could",
        "really",
        "just",
        "product",
        "purchase",
        "review"
    }


    words = [

        w for w in words

        if w not in stopwords
    ]


    word_counts = (

        pd.Series(words)

        .value_counts()

        .head(10)
    )


    if len(word_counts):

        keyword_items = [

            html.Li(
                f"{word}: {count}"
            )

            for word, count
            in word_counts.items()
        ]

    else:

        keyword_items = [

            html.Li(
                "No negative keywords found."
            )
        ]


    # --------------------------------------------------------
    # NEGATIVE REVIEWS
    # --------------------------------------------------------

    negative_df = df[
        df["Sentiment"]
        == "NEGATIVE"
    ]


    if negative_df.empty:

        negative_items = [

            html.P(
                "No negative reviews found."
            )
        ]

    else:

        negative_items = [

            html.Div(

                [

                    html.B(
                        f"Review "
                        f"{row['Review Number']}: "
                    ),

                    html.Span(
                        row["Review"]
                    )

                ],

                style={

                    "marginBottom":
                        "12px",

                    "padding":
                        "10px",

                    "border":
                        "1px solid #ddd",

                    "borderRadius":
                        "6px"
                }
            )

            for _, row
            in negative_df.iterrows()
        ]


    # --------------------------------------------------------
    # RETURN DASHBOARD DATA
    # --------------------------------------------------------

    return (

        len(df),

        int(
            counts["POSITIVE"]
        ),

        int(
            counts["NEGATIVE"]
        ),

        int(
            counts["NEUTRAL"]
        ),

        pie,

        trend_fig,

        count_fig,

        keyword_items,

        negative_items,

        "/assets/negative_wordcloud.png"
    )


# ============================================================
# RUN DASHBOARD
# ============================================================

if __name__ == "__main__":

    print(
        "Starting Smart Review Analytics Dashboard..."
    )

    print(
        "Using:",
        DATA_FILE
    )

    print(
        "Open: http://127.0.0.1:8050"
    )

    app.run(
        host="127.0.0.1",
        port=8050,
        debug=False
    )