import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from bertopic import BERTopic

# Load the data
posts_df = pd.read_csv("fact_posts.csv")
topics_df = pd.read_csv("dim_topic.csv")
topic_model = BERTopic.load("./model")

# Data Preparation
posts_df['date'] = pd.to_datetime(posts_df['date'])
df = pd.merge(posts_df, topics_df, on='topic_cluster', how='left')
df.rename(columns={'date': 'Date', 'content': 'Content', 'author': 'Author'}, inplace=True) # Consistent naming

st.title("Prediction Market Discord Message Analysis")
st.text(f"Chats in Polymarket and Kalshi are scraped from {posts_df.date.min()} to {posts_df.date.max()}. Chats are cleaned and clustered with BERTopic")

# Sidebar Filters
st.sidebar.header("Filters")

# --- Protocol Filter ---
protocols = st.sidebar.multiselect("Select Protocol", df['protocol'].unique(), default=df['protocol'].unique())

# --- Boolean Filters ---
st.sidebar.subheader("Topic Type Filters")
boolean_filters = ['is_product_complaint', 'is_product_feedback', 'is_product_competitor', 'is_bet_discussions']
boolean_filter_values = {}
for bool_filter in boolean_filters:
    boolean_filter_values[bool_filter] = st.sidebar.checkbox(bool_filter.replace('_', ' ').title())

# --- Date Range Filter ---
st.sidebar.subheader("Date Range")
date_range = st.sidebar.date_input("Select Date Range",
                                   min_value=df['Date'].min().date(),
                                   max_value=df['Date'].max().date(),
                                   value=(df['Date'].min().date(), df['Date'].max().date()))

# Apply Filters
filtered_df = df[df['protocol'].isin(protocols)]

# Apply Boolean Filters
for bool_filter, value in boolean_filter_values.items():
    if value:
        filtered_df = filtered_df[filtered_df[bool_filter] == value]

filtered_df = filtered_df[(filtered_df['Date'].dt.date >= date_range[0]) & (filtered_df['Date'].dt.date <= date_range[1])]

# --- Topic Ranking Table ---
st.header("Topic Overview")

topic_counts_overview = filtered_df['topic_name'].value_counts().reset_index()
topic_counts_overview.columns = ['topic_name', 'message_count']
topic_overview = pd.merge(topic_counts_overview, topics_df[['topic_name', 'representation']], on='topic_name', how='left')
topic_overview = topic_overview.sort_values(by='message_count', ascending=False)

st.subheader("Topics Ranked by Size")
st.dataframe(topic_overview[['topic_name', 'message_count', 'representation']], use_container_width=True)

# Ranked Bar Chart
fig_topic_overview = px.bar(topic_overview, x='topic_name', y='message_count',
                            labels={'message_count': 'Number of Messages', 'topic_name': 'Topic'},
                            title='Topic Popularity',
                            )
st.plotly_chart(fig_topic_overview)

# --- Section 1: Sentiment Analysis ---
st.header("Sentiment Analysis")
topic_options_timeseries = filtered_df['topic_name'].dropna().unique()
default_topic = topic_overview['topic_name'].iloc[0] if not topic_overview.empty else None
selected_topic_time_series = st.selectbox("Select a Topic for Time Series", options=topic_options_timeseries, index=list(topic_options_timeseries).index(default_topic) if default_topic in topic_options_timeseries else 0)

st.subheader("Sentiment Over Time")

if selected_topic_time_series:
    # Filter data for the selected topic
    topic_time_series_df = filtered_df[filtered_df['topic_name'] == selected_topic_time_series]

    # Aggregate sentiment by month
    sentiment_over_time = topic_time_series_df.groupby(pd.Grouper(key='Date', freq='M'))['sentiment_score'].mean().reset_index()

    if not sentiment_over_time.empty:
        # Line chart for sentiment trend
        fig_sentiment_trend = px.line(sentiment_over_time, x='Date', y='sentiment_score',
                                     labels={'sentiment_score': 'Average Sentiment', 'Date': 'Month'},
                                     title=f'Sentiment Trend for {selected_topic_time_series}')
        st.plotly_chart(fig_sentiment_trend)

        # Display messages sorted by sentiment (most unhappy first)
        st.subheader(f"Messages for {selected_topic_time_series} (Sorted by Most Negative Sentiment)")
        sorted_messages = topic_time_series_df.sort_values(by='sentiment_score', ascending=True)
        st.dataframe(sorted_messages[['Author', 'Date', 'sentiment_score', 'Content']].rename(columns={'sentiment_score': 'Sentiment Score'}), use_container_width=True)
    else:
        st.info("No data available for the selected topic in the current filter.")

topic_options_search = filtered_df['topic_name'].dropna().unique()
# Removed the dropdown filter under search by topic
# selected_topic_search = st.selectbox("Select a Topic", options=[''] + list(topic_options_search), key='topic_search')
selected_topic_search = None # Initialize to None since the dropdown is removed

# --- Section 3: Good vs. Bad Insights ---
st.header("Positive vs. Negative Sentiment Examples")

if selected_topic_time_series: # Use the topic filter from sentiment analysis
    topic_filtered_df = filtered_df[filtered_df['topic_name'] == selected_topic_time_series]

    # --- Good Messages ---
    st.subheader(f"Positive Sentiment Messages for {selected_topic_time_series}")
    positive_messages = topic_filtered_df[topic_filtered_df['sentiment_score'] > 0].sort_values(by='sentiment_score', ascending=False).reset_index(drop=True)
    st.write(f"Number of positive messages: {len(positive_messages)}")
    if not positive_messages.empty:
        st.dataframe(positive_messages[['Author', 'Date', 'sentiment_score', 'topic_name', 'Content']].rename(columns={'sentiment_score': 'Sentiment'}), use_container_width=True)

    # --- Bad Messages ---
    st.subheader(f"Negative Sentiment Messages for {selected_topic_time_series}")
    negative_messages = topic_filtered_df[topic_filtered_df['sentiment_score'] < 0].sort_values(by='sentiment_score', ascending=True).reset_index(drop=True)
    st.write(f"Number of negative messages: {len(negative_messages)}")
    if not negative_messages.empty:
        st.dataframe(negative_messages[['Author', 'Date', 'sentiment_score', 'topic_name', 'Content']].rename(columns={'sentiment_score': 'Sentiment'}), use_container_width=True)
else:
    st.info("Select a topic under 'Sentiment Analysis' to see positive and negative sentiment messages.")

# --- Section 4: Interactive Data Exploration ---
st.header("Interactive Data Exploration")

st.subheader("Explore Data with Pandas Pivot Table")
index_col = st.selectbox("Select Index Column for Pivot Table", filtered_df.columns.tolist(), key='index_col')
columns_col = st.selectbox("Select Column for Pivot Table", [None] + filtered_df.columns.tolist(), key='columns_col')
values_col = st.selectbox("Select Value Column for Pivot Table", filtered_df.select_dtypes(include=['number']).columns.tolist(), key='values_col')
aggfunc = st.selectbox("Select Aggregation Function", ['mean', 'sum', 'count', 'median', 'min', 'max'], key='aggfunc')

if values_col:
    pivot_interactive = filtered_df.pivot_table(index=index_col, columns=columns_col, values=values_col, aggfunc=aggfunc)
    st.dataframe(pivot_interactive, use_container_width=True)


st.header("BERTopic Intertopic Distance Map")
# Generate the intertopic distance map using BERTopic's visualize_topics method
# This method returns a plotly.graph_objects.Figure object
try:
    fig = topic_model.visualize_topics()
    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Error generating or displaying the intertopic distance map: {e}")

st.write("This visualization shows the distance between different topics. Topics that are closer together are more similar.")


st.header("Similarity Map")
# Generate the intertopic distance map using BERTopic's visualize_topics method
# This method returns a plotly.graph_objects.Figure object
try:
    fig = topic_model.visualize_heatmap()
    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Error generating or displaying the similarity map: {e}")

st.write("This visualization shows the similarity in topics. This is to find distinctively different topics")



st.header("Topic Hierarchy")
# Generate the intertopic distance map using BERTopic's visualize_topics method
# This method returns a plotly.graph_objects.Figure object
try:
    fig = topic_model.visualize_hierarchy()
    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Error generating or displaying the hierarchy map: {e}")

st.write("This graph continues to group the topics together so that we can implicitly find the parent topic")




