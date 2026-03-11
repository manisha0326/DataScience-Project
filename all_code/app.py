import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import joblib
import dash_bootstrap_components as dbc

# Load model and vectorizer
model = joblib.load("../dataset/mood_model.pkl")
vectorizer = joblib.load("../dataset/vectorizer.pkl")

# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1(
                    "Human Mood Prediction",
                    className="text-center text-primary mb-4",
                    style={"margin-top": "40px"},
                ),
                width=12,
            )
        ),
        dbc.Row(
            dbc.Col(
                [
                    html.Div(
                        id="chat-box",
                        style={
                            "border": "1px solid #ddd",
                            "padding": "10px",
                            "height": "400px",
                            "overflowY": "auto",
                            "backgroundColor": "#f9f9f9",
                            "borderRadius": "10px",
                            "marginBottom": "10px",
                        },
                    ),
                    # Input box (Enter works now)
                    dcc.Input(
                        id="user-input",
                        type="text",
                        placeholder="Type how you feel...",
                        n_submit=0,
                        style={
                            "width": "100%",
                            "height": "50px",
                            "borderRadius": "5px",
                            "padding": "10px",
                        },
                    ),
                    html.Br(),
                    dbc.Button(
                        "Submit",
                        id="send-btn",
                        n_clicks=0,
                        className="mt-2",
                        style={
                            "backgroundColor": "#007BFF",
                            "color": "white",
                            "border": "1px solid #007BFF",
                        },
                    ),
                ],
                width=8,
                className="mx-auto",
            )
        ),
    ],
    fluid=True,
)


# Callback (Button OR Enter will submit)
@app.callback(
    Output("chat-box", "children"),
    Output("user-input", "value"),
    Input("send-btn", "n_clicks"),
    Input("user-input", "n_submit"),
    State("user-input", "value"),
    State("chat-box", "children"),
)
def update_chat(btn_clicks, enter_press, user_text, chat_children):

    if user_text:

        vect_text = vectorizer.transform([user_text])
        prediction = model.predict(vect_text)[0]

        user_msg = html.Div(
            [html.B("You: "), html.Span(user_text)],
            style={"margin": "5px 0", "textAlign": "left"},
        )

        bot_msg = html.Div(
            [html.Span(f"Predicted Mood → {prediction}")],
            style={"margin": "5px 0", "textAlign": "right", "color": "green"},
        )

        if chat_children is None:
            chat_children = []

        chat_children = chat_children + [user_msg, bot_msg]

        return chat_children, ""

    return chat_children, ""


if __name__ == "__main__":
    app.run(debug=True)
