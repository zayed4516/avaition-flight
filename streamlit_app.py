import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Setting page configuration
st.set_page_config(page_title="Aviation Flights Fare", page_icon="✈️", layout='wide')

# Loading data
df = pd.read_csv('cleaned_df.csv')

with st.sidebar:
    st.sidebar.image('R.jpg')
    st.sidebar.subheader("This dashboard for Indian Aviation Flights Fare aimed at predicting the prices of flight tickets")
    st.sidebar.write("")

    data = df.copy()
    source = st.sidebar.selectbox("Departure City", ['All'] + list(data['Source'].unique()))
    if source != 'All':
        data = data[data['Source'] == source]

    destination = st.sidebar.selectbox("Arrival City", ['All'] + list(data['Destination'].unique()))
    if destination != 'All':
        data = data[data['Destination'] == destination]

    airline = st.sidebar.selectbox("Airline Carrier", ['All'] + list(data['Airline'].unique()))
    if airline != 'All':
        data = data[data['Airline'] == airline]

    add_info = st.sidebar.selectbox("Additional Services", ['All'] + list(data['Additional_Info'].unique()))
    filter_box = st.sidebar.selectbox("Filter Prices on", [None, 'Day', 'Month', 'Dep_Hour'])

    st.sidebar.write("")
    st.sidebar.markdown("Made by [Hussein Zayed](https://github.com/HusseinZayed)")

# Filtering Function
def filter(airline, source, destination, add_info):
    if airline == 'All' and source == 'All' and destination == 'All' and add_info == 'All':
        filtered_data = data.copy()
    else:
        filtered_data = data

        if source != 'All':
            filtered_data = filtered_data[filtered_data['Source'] == source]

        if destination != 'All':
            filtered_data = filtered_data[filtered_data['Destination'] == destination]

        if airline != 'All':
            filtered_data = filtered_data[filtered_data['Airline'] == airline]

        if add_info != 'All':
            filtered_data = filtered_data[filtered_data['Additional_Info'] == add_info]

    return filtered_data

# Dashboard Tabs
tab1, tab2, tab3 = st.tabs(["🏠 Home", "📈 Insights", "🤖 Prediction"])

# Home Page
with tab1:
    st.write("If you are a traveler looking to plan your next trip, or you are an airline or travel agency, "
             "you need to know about ticket and service price variations.\n"
             "Airline ticket pricing has become increasingly complex due to factors such as demand fluctuations and seasonal trends.\n"
             "\n"
             "My project aims to help you make the right decision and buy the best ticket at the best price by developing a predictive model "
             "that can accurately estimate flight fares based on the given features.")

    im1 = Image.open('R.jpg')
    st.image(im1)

# Insights Page
with tab2:
    filtered_data = filter(airline, source, destination, add_info)

    card1, card2, card3, card4 = st.columns((2, 2, 2, 4))
    flight_count = filtered_data['Airline'].count()
    highest_price = filtered_data['Price'].max()
    lowest_price = filtered_data['Price'].min()
    top_airline = filtered_data['Airline'].value_counts().idxmax()
    card1.metric("Flight Count", f"{flight_count}")
    card2.metric("Highest Price", f"{highest_price}")
    card3.metric("Lowest Price", f"{lowest_price}")
    card4.metric("Top Airline", f"{top_airline}")

    visual1, visual2 = st.columns((5, 5))
    with visual1:
        st.subheader('Top Airlines')
        most_airline = filtered_data['Airline'].value_counts().sort_values(ascending=False).head()
        fig = px.pie(data_frame=most_airline, 
                     names=most_airline.index, 
                     values=most_airline.values, 
                     hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    top_airlines = most_airline.index.tolist()
    with visual2:
        st.subheader('Airline Price')
        airline_price = filtered_data[filtered_data['Airline'].isin(top_airlines)].groupby('Airline')['Price'].min().sort_values(ascending=False)
        fig = px.bar(airline_price, 
                    x=airline_price.index, 
                    y=airline_price.values)
        fig.update_xaxes(title='Airline')
        fig.update_yaxes(title='Price')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader('Duration vs Price')
    fig = px.scatter(filtered_data,
                    x='Duration',
                    y='Price',
                    color=filter_box)
    fig.update_xaxes(title='Duration')
    fig.update_yaxes(title='Price')
    fig.update_layout(
        legend=dict(
            borderwidth=2,
            orientation='h',
            x=1,
            y=1,
            xanchor='right',
            yanchor='top',
            traceorder='normal'
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("""From the initial inspection of the dataset, here are some key variables present:

1. **Airline**: The carrier for each flight.
2. **Source**: The departure city.
3. **Destination**: The arrival city.
4. **Duration**: The total flight duration (in minutes).
5. **Total_Stops**: The number of stops during the flight.
6. **Additional_Info**: Any extra information regarding the flight.
7. **Price**: The price of the flight.
8. **Day**: The day of the month.
9. **Month**: The month of the year.
10. **Dep_Hour**: The departure hour.

I will now perform some exploratory data analysis (EDA) to find deeper insights into this dataset, such as relationships between price and other factors, popular airlines, and more.

Here are some key insights from the exploratory data analysis (EDA) based on the summary statistics:

1. **Flight Duration**:
   - The average flight duration is approximately 630 minutes (around 10.5 hours).
   - The shortest flight duration is 5 minutes, and the longest is 2860 minutes (about 47.5 hours), indicating a wide range of flight durations.

2. **Total Stops**:
   - The average number of stops per flight is around 0.8, indicating that most flights are direct or have one stop.
   - Some flights have up to 4 stops, while others are direct with 0 stops.

3. **Price**:
   - The average flight price is 9027 units.
   - The cheapest flight is priced at 1759 units, while the most expensive flight costs 79,512 units.
   - There is significant variation in flight prices, suggesting that factors such as airline, duration, and number of stops may heavily influence the price.

4. **Day of Travel**:
   - Flights are spread across different days of the month, with no specific pattern based on the day alone, as the average day is around the 13th.

5. **Month of Travel**:
   - The data covers flights mainly from March to June.

6. **Departure Hour**:
   - The average departure time is around 12:45 PM.
   - The earliest departure is at midnight (0:00), and the latest is at 11:00 PM (23:00).

Next, I'll further analyze relationships between price and other factors (e.g., total stops, airline, and duration) to see what drives the price variations.

Here are the insights from the visualizations:""")

    st.write('''1. **Average Price vs. Total Stops**:
   - Flights with more stops tend to have higher average prices, especially for flights with 2 or more stops. This suggests that additional stops increase the cost of the flight.''')

    plt.figure(figsize=(20,10))
    sns.barplot(data=df.sort_values('Price', ascending=False), x='Total_Stops', y='Price')
    plt.xlabel('Total Stops')
    plt.ylabel('Mean Price')
    plt.title('Total Stops Mean Price')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    st.write("2. **Price vs. Duration**: - There is a noticeable positive correlation between flight duration and price. Generally, longer flights tend to be more expensive, but the relationship is not linear, indicating that other factors (e.g., the airline or number of stops) might also influence prices.")
    plt.figure(figsize=(20,10))
    sns.scatterplot(data=df, x='Duration', y='Price')
    plt.xlabel('Duration')
    plt.ylabel('Mean Price')
    plt.title('Duration Mean Price')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    st.write("Here are the deeper insights from the correlation heatmap:")
    plt.figure(figsize=(14, 6))
    sns.heatmap(df[['Duration', 'Total_Stops', 'Price', 'Dep_Hour']].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap of Numerical Variables')
    st.pyplot(plt)

# Prediction Page
with tab3:
    st.write("""Here are some additional insights and predictions:

7. **Prediction Page**
    - This page allows users to input parameters and get predictions on flight prices.
    - The predictive model uses features such as Departure City, Arrival City, Airline, Duration, Total Stops, and more.

    Please enter the details below to get a flight fare prediction.""")

    with st.form(key='prediction_form'):
        st.subheader('Flight Prediction Form')

        # User inputs
        dep_city = st.selectbox("Departure City", ['Banglore', 'Chennai', 'Delhi', 'Kolkata', 'Mumbai'])
        arr_city = st.selectbox("Arrival City", ['Banglore', 'Chennai', 'Delhi', 'Kolkata', 'Mumbai'])
        airline = st.selectbox("Airline", ['Air India', 'GoAir', 'IndiGo', 'Jet Airways', 'SpiceJet'])
        duration = st.number_input("Duration (in minutes)", min_value=5, max_value=3000, value=120)
        stops = st.selectbox("Total Stops", [0, 1, 2, 3, 4])
        dep_hour = st.number_input("Departure Hour (0-23)", min_value=0, max_value=23, value=12)

        submit_button = st.form_submit_button(label='Predict Price')

        if submit_button:
            # Load the trained model
            model = joblib.load('flight_price_model.pkl')
            
            # Prepare the input data
            input_data = pd.DataFrame({
                'Source': [dep_city],
                'Destination': [arr_city],
                'Airline': [airline],
                'Duration': [duration],
                'Total_Stops': [stops],
                'Dep_Hour': [dep_hour]
            })

            # Apply any necessary preprocessing (e.g., encoding categorical features)
            input_data = pd.get_dummies(input_data)

            # Make the prediction
            prediction = model.predict(input_data)

            st.write(f"The predicted flight price is: {prediction[0]:,.2f}")
