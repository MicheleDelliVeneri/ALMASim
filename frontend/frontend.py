import dash
from dash import dcc, html, Input, Output, State
import requests
import pandas as pd

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # Use Dash server instance for deployment

# Adjust if running backend on a different host/port
API_URL = "http://localhost:8000"

# App layout
app.layout = html.Div(
    [
        html.H1("ALMASim Dashboard"),
        # Science Keywords and Categories
        html.Div(
            [
                html.H3("Fetch Science Keywords and Categories"),
                html.Button(
                    "Fetch Science Keywords", id="fetch_science_keywords", n_clicks=0
                ),
                html.Div(id="science_keywords_output"),
            ],
            style={"margin-bottom": "30px"},
        ),
        # Target-Based Query
        html.Div(
            [
                html.H3("Query by Targets"),
                dcc.Textarea(
                    id="targets_input",
                    placeholder="Enter targets as JSON, e.g., "
                    '[{"target_name": "NGC253", "member_ous_uid": \
                        "uid://A001/X122/X1"}]',
                    style={"width": "100%", "height": "100px"},
                ),
                html.Button("Query Targets", id="query_targets_button", n_clicks=0),
                html.Div(id="targets_output"),
            ],
            style={"margin-bottom": "30px"},
        ),
        # Science-Based Query
        html.Div(
            [
                html.H3("Query by Science Filters"),
                html.Div(
                    [
                        html.Label("Science Keywords (comma-separated):"),
                        dcc.Input(
                            id="science_keywords_input",
                            type="text",
                            placeholder="e.g., Star Formation,AGN",
                        ),
                        html.Label("Scientific Categories (comma-separated):"),
                        dcc.Input(
                            id="scientific_categories_input",
                            type="text",
                            placeholder="e.g., ISM,Extragalactic",
                        ),
                        html.Label("Bands (comma-separated):"),
                        dcc.Input(
                            id="bands_input", type="text", placeholder="e.g., 3,6,7"
                        ),
                        html.Label("Frequency Range (min,max in GHz):"),
                        dcc.Input(
                            id="frequency_range_input",
                            type="text",
                            placeholder="e.g., 100,300",
                        ),
                        html.Label("Spatial Resolution Range (min,max in arcsec):"),
                        dcc.Input(
                            id="spatial_resolution_input",
                            type="text",
                            placeholder="e.g., 0.1,1.0",
                        ),
                        html.Label("Velocity Resolution Range (min,max in km/s):"),
                        dcc.Input(
                            id="velocity_resolution_input",
                            type="text",
                            placeholder="e.g., 5,20",
                        ),
                    ],
                    style={"margin-bottom": "20px"},
                ),
                html.Button("Query Science", id="query_science_button", n_clicks=0),
                html.Div(id="science_output"),
            ]
        ),
    ]
)


# Callbacks
@app.callback(
    Output("science_keywords_output", "children"),
    Input("fetch_science_keywords", "n_clicks"),
)
def fetch_science_keywords(n_clicks):
    if n_clicks > 0:
        try:
            response = requests.get(f"{API_URL}/science_keywords/")
            if response.status_code == 200:
                data = response.json()
                return html.Div(
                    [
                        html.H4("Science Keywords:"),
                        html.Ul([html.Li(kw) for kw in data["science_keywords"]]),
                        html.H4("Scientific Categories:"),
                        html.Ul(
                            [html.Li(cat) for cat in data["scientific_categories"]]
                        ),
                    ]
                )
            else:
                return f"Error fetching science keywords: \
                        {response.status_code}"
        except Exception as e:
            return f"Error: {e}"
    return ""


@app.callback(
    Output("targets_output", "children"),
    Input("query_targets_button", "n_clicks"),
    State("targets_input", "value"),
)
def query_targets(n_clicks, targets_input):
    if n_clicks > 0 and targets_input:
        try:
            # Use eval cautiously; prefer json.loads
            targets = eval(targets_input)
            response = requests.post(f"{API_URL}/query_targets/", json=targets)
            if response.status_code == 200:
                data = response.json()
                if "message" in data:
                    return data["message"]
                df = pd.DataFrame(data)
                return html.Div(
                    [
                        html.H4("Query Results:"),
                        dcc.Graph(
                            figure={
                                "data": [
                                    {
                                        "x": df["target_name"],
                                        "y": df["frequency"],
                                        "type": "bar",
                                        "name": "Frequency",
                                    }
                                ],
                                "layout": {"title": "Target Frequencies"},
                            }
                        ),
                    ]
                )
            else:
                return f"Error querying targets: {response.status_code}"
        except Exception as e:
            return f"Error: {e}"
    return ""


@app.callback(
    Output("science_output", "children"),
    Input("query_science_button", "n_clicks"),
    State("science_keywords_input", "value"),
    State("scientific_categories_input", "value"),
    State("bands_input", "value"),
    State("frequency_range_input", "value"),
    State("spatial_resolution_input", "value"),
    State("velocity_resolution_input", "value"),
)
def query_science(
    n_clicks,
    science_keywords,
    scientific_categories,
    bands,
    frequency_range,
    spatial_resolution,
    velocity_resolution,
):
    if n_clicks > 0:
        try:
            filters = {
                "science_keywords": (
                    [kw.strip() for kw in science_keywords.split(",")]
                    if science_keywords
                    else None
                ),
                "scientific_categories": (
                    [cat.strip() for cat in scientific_categories.split(",")]
                    if scientific_categories
                    else None
                ),
                "bands": [int(b) for b in bands.split(",")] if bands else None,
                "frequency": (
                    tuple(map(float, frequency_range.split(",")))
                    if frequency_range
                    else None
                ),
                "spatial_resolution": (
                    tuple(map(float, spatial_resolution.split(",")))
                    if spatial_resolution
                    else None
                ),
                "velocity_resolution": (
                    tuple(map(float, velocity_resolution.split(",")))
                    if velocity_resolution
                    else None
                ),
            }
            response = requests.post(f"{API_URL}/query_science/", json=filters)
            if response.status_code == 200:
                data = response.json()
                if "message" in data:
                    return data["message"]
                df = pd.DataFrame(data)
                return html.Div(
                    [
                        html.H4("Query Results:"),
                        dcc.Graph(
                            figure={
                                "data": [
                                    {
                                        "x": df["target_name"],
                                        "y": df["frequency"],
                                        "type": "bar",
                                        "name": "Frequency",
                                    }
                                ],
                                "layout": {"title": "Science Filter Results"},
                            }
                        ),
                    ]
                )
            else:
                return f"Error querying science filters: \
                    {response.status_code}"
        except Exception as e:
            return f"Error: {e}"
    return ""


# Run Dash app
if __name__ == "__main__":
    app.run_server(debug=True)
