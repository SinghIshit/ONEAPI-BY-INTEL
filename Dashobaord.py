import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Set page title and favicon
st.set_page_config(page_title="Interactive Dashboard with Streamlit", page_icon=":chart_with_upwards_trend:")

# Set page heading
st.title("Interactive Data Visualizer")

# Create a file upload box for CSV files
file = st.file_uploader("Upload a CSV file", type=["csv"])

# Create a checkbox to display the raw data
show_raw_data = st.checkbox("Show raw data")

if file is not None:
    # Load the data into a Pandas DataFrame
    df = pd.read_csv(file)

    # Create a dropdown menu to select a column for grouping
    group_column = st.selectbox("Group by", df.columns,key='1')
    group_column1 = st.selectbox("Group by", df.columns,key='2')

    # Create a bar chart of the selected column
    fig = px.bar(df, x=group_column1, y=group_column)

    # Display the chart
    st.plotly_chart(fig)

    # Display the raw data if the checkbox is checked
    if show_raw_data:
        st.write(df)
