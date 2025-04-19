import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px         
import pickle
from prophet import Prophet
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from models.prophet_model import train_prophet_model, forecast_aqi
def set_background(image_url):

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
def set_text_color():
    st.markdown(
        """
        <style>
        /* Change the text color to white */
        .stTitle, .stHeader, .stText, .stMarkdown, .stSubHeader {
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def welcome_page():

    set_background("https://i.imgur.com/w9qgmgO.jpeg")  #https://i.imgur.com/LuV9Pw2.jpeg
    set_text_color()
    st.title("Welcome to the Air Quality Prediction and Forecasting App!")
    
    st.write("""
        This web app provides real-time prediction and forecasting of the Air Quality Index (AQI) 
        for Kathmandu using machine learning models and the Prophet model for forecasting.
        It allows users to visualize historical AQI data, predict AQI based on environmental parameters,
        and forecast future AQI for the selected duration.
    """)
    
    st.write("### Group Members:")
    st.write("- Sangam Paudel")
    st.write("- Saroj Rawal")
    st.write("- Subesh Yadav")
    
    # Button to enter dashboard
    if st.button("Enter Dashboard"):
           st.session_state.entered_dashboard = True
           st.markdown("✅ Click again to confirm.")
           return True
    else:
           return False

def main():
    # Initialize session state if it doesn't exist
    if "entered_dashboard" not in st.session_state:
        st.session_state.entered_dashboard = False
    
    # If the user hasn't entered the dashboard yet, show the welcome page
    if not st.session_state.entered_dashboard:
        if welcome_page():
            return  # Stop further code execution and stay on the welcome page
    
    else:
# ........................................... Load Model & Data ..........................................
        st.set_page_config(layout="wide")

        # Update paths to use relative paths for Streamlit Cloud compatibility
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'processed_data.csv')

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        @st.cache_data
        def load_data():
            try:
                return pd.read_csv(data_path, parse_dates=["Datetime"])
            except FileNotFoundError:
                return pd.DataFrame(columns=["Datetime", "AQI"])
        
        st.sidebar.title("Air Quality Dashboard")
        selection = st.sidebar.radio("Select Option", ["Visualize", "Predict", "Forecast"])
        
#............................................ Visualize Page ..............................................

        if selection == "Visualize":
            
            st.title("🔎 Air Quality Visualization")
            st.write("Here, We can see the historical air quality data (AQI) Visualization.")
        
            df = load_data()
            if not df.empty:
                tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Data Table", "📈 Time Series", "📊 Statistics", "📉 Trend"])
                with tab1:
                    st.write("### Historical AQI Data of Kathmandu")
                    search_term = st.text_input("Search data with Date:", placeholder="YYYY-MM-DD")
                    if search_term:
                        filtered_data = df[df["Datetime"].dt.strftime("%Y-%m-%d").str.contains(search_term)]
                    else:
                       filtered_data = df
                    st.dataframe(filtered_data)

                    st.write("### Correlation among the features")
                    # Drop the 'Datetime' column before computing the correlation matrix
                    df_numeric = df.drop(columns=['Datetime'])
                    plt.figure(figsize=(10, 6))
                    sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
                    plt.title("Correlation Heatmap")
                    st.pyplot(plt)

                with tab2:    
                # Plot AQI over time
                    st.write("#### You can choose the date where you want to Visualize the AQI trends")
                    start_date = st.date_input("Start Date", 
                        value=df["Datetime"].min().date(),
                        min_value=df["Datetime"].min().date(), 
                        max_value=df["Datetime"].max().date())
                    
                    end_date = st.date_input("End Date", 
                        value=df["Datetime"].max().date(), 
                        min_value=df["Datetime"].min().date(), 
                        max_value=df["Datetime"].max().date())
                    
                    filtered_data = df[(df["Datetime"].dt.date >= start_date) & 
                                    (df["Datetime"].dt.date <= end_date)]
                    
                    time_fig = px.line(
                    filtered_data, 
                    x="Datetime", 
                    y="AQI",
                    title="Air Quality Index Over Time",
                    labels={"AQI": "AQI Value"},
                    line_shape="spline",
                    template="plotly_white",
                    height=500,
                    
                    )
                    st.plotly_chart(time_fig, use_container_width=True)
    
                    if len(filtered_data) > 30:  # Only show if we have enough data
                        filtered_data['YearMonth'] = filtered_data['Datetime'].dt.to_period('M')
                        monthly_avg = filtered_data.groupby('YearMonth')['AQI'].mean().reset_index()
                        monthly_avg['YearMonth'] = monthly_avg['YearMonth'].astype(str)
                        
                        bar_fig = px.bar(
                            monthly_avg, 
                            x='YearMonth', 
                            y='AQI',
                            title='Monthly Average AQI',
                            labels={'AQI': 'Average AQI', 'YearMonth': 'Month'},
                            color='AQI',
                            color_continuous_scale=px.colors.sequential.Viridis,
                            template="plotly_white"
                        )
                        st.plotly_chart(bar_fig, use_container_width=True)
                
                with tab3:
                    st.subheader("AQI Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Average AQI", f"{df['AQI'].mean():.1f}")
                    col2.metric("Maximum AQI", f"{df['AQI'].max():.1f}")
                    col3.metric("Minimum AQI", f"{df['AQI'].min():.1f}")
                    col4.metric("Total Data Points", len(df))

                    
                    fig = px.histogram(
                          df, 
                          x="AQI",
                          nbins=30,
                          title="AQI Frequency Distribution",
                          labels={"AQI": "Air Quality Index", "count": "Frequency"},
                          color_discrete_sequence=["#1E3A8A"],
                          template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.sidebar.markdown(" ")
                    st.write("### Distribution of AQI categories by Pie-Chart")
                    def categorize_aqi_simple(aqi):
                        if aqi <= 50:
                            return "Good(<50)"
                        elif aqi <= 100:
                            return "Moderate(50<100)"
                        elif aqi <= 150:
                            return "Unhealthy for Sensitive Groups(100<150)"
                        elif aqi <= 200:
                            return "Unhealthy(150<200)"
                        elif aqi <= 300:
                            return "Very Unhealthy(200<300)"
                        else:
                            return "Hazardous(>300)"
        
        
                    df['AQI_Category'] = df['AQI'].apply(categorize_aqi_simple)
                        
                    category_counts = df['AQI_Category'].value_counts().reset_index()
                    category_counts.columns = ['Category', 'Count']
            
                    fig = px.pie(
                          category_counts, 
                          values='Count', 
                          names='Category',
                          color='Category',
                          hole=0.4,
                          template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                with tab4:
                    st.subheader("Yearly,weekly and Dailly Trends")
                    trends_clicked = st.button("Show Trends", type="primary")
                    if trends_clicked:
                        with st.spinner("Generating....."):
                           pmodel = train_prophet_model(df)
                           forecast = forecast_aqi(pmodel)
                           fig = pmodel.plot_components(forecast)
                           st.pyplot(fig)
            else:
                st.warning("No data available for visualization.")
        
        
# ............................................. Predict Page .....................................................

        elif selection == "Predict":
            st.title("🔮 Predict Air Quality")
            
            col1, col2 = st.columns([4, 2])
            with col1:
                st.subheader("Input Parameters")
                st.write("Enter the environmental parameters below to predict the Air Quality Index (AQI).The model will use these values to estimate the current air quality level.")
                st.markdown("**Particulate matter**")
                # User input form 
                col_pm1, col_pm2 = st.columns(2)
                with col_pm1:
                   pm25 = st.number_input("PM2.5 (μg/m³)", min_value=0.0, value=20.3, format="%.2f",help="Fine particulate matter with diameter less than 2.5 micrometers")
                with col_pm2: 
                   pm10 = st.number_input("PM10 (μg/m³)", min_value=0.0, value=30.4, format="%.2f",help="Particulate matter with diameter less than 10 micrometers")
                
                st.markdown("**Gases**")
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                   co = st.number_input("CO (μg/m³)", min_value=0.0, value=0.5, format="%.2f",help="Carbon Monoxide concentration")
                   no2 = st.number_input("NO2 (μg/m³)", min_value=0.0, value=10.3, format="%.2f",help="Nitrogen Dioxide concentration")
                with col_g2:
                   so2 = st.number_input("SO2 (μg/m³)", min_value=0.0, value=5.0, format="%.2f",help="Sulfur Dioxide concentration")
                   o3 = st.number_input("O3 (μg/m³)", min_value=0.0, value=30.0, format="%.2f",help="Ozone concentration")
                
                st.markdown("**Weather**")
                col_w1, col_w2 = st.columns(2)
                with col_w1:
                   temp = st.number_input("Temperature (°C)", min_value=-10.0, value=25.2, format="%.2f",help="Temperature in degrees Celsius")
                   humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, format="%.2f",help="Relative humidity percentage")
                with col_w2:
                    wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, value=10.0, format="%.2f",help="Wind speed in kilometers per hour")
                predict_clicked = st.button("Predict", type="primary", use_container_width=True)
            
            with col2:
                st.markdown("<h3 style='text-align: center;'>Prediction Result</h3>", unsafe_allow_html=True)

                def categorize_aqi(aqi):
                    if aqi <= 50:
                        return "Good, Air Quality!", "green", "You can go outside and Enjoy your day!", "😊"
                    elif aqi <= 100:
                        return "Moderate.. Air Quality!", "yellow", "Reduce outdoor exercises!", "😐"
                    elif aqi <= 150:
                        return "Unhealthy for Sensitive Groups....", "orange", "Wear a mask outdoors.\n  Close windows to avoid dirty air", "😷"
                    elif aqi <= 200:
                        return "Unhealthy ! Everyone may experience health effects.", "red", "Avoid outdoor activities.\n Close your house window", "😷"
                    elif aqi <= 300:
                        return "Very Unhealthy !! Serious health risks.", "purple", "Stay indoors and use an air purifier if possible.", "😷"
                    else:
                        return "Hazardous !!!!  Health warning.", "maroon", "Avoid going outside, wear a high-quality mask.", "😷"
        
            # Predict button
                if predict_clicked:
                    try:
                         input_data = np.array([[pm25, pm10, co, no2, so2, o3, temp, humidity, wind_speed]])
                         
                         predicted_aqi = model.predict(input_data)[0]
                         
                         # Get AQI category
                         category, color, advice, emoji = categorize_aqi(predicted_aqi)
                
                # Display prediction with improved styling
                         st.markdown(
                                 f"""
                                    <div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px; text-align: center;">
                                        <h3 style="color: #333;">Predicted AQI</h3>
                                        <p style="font-size: 48px; font-weight: bold; color: {color};">{predicted_aqi:.1f}</p>
                                        <p style="background-color: {color}; color: white; padding: 10px; border-radius: 5px; display: inline-block;">
                                            {category} {emoji}
                                        </p>
                                        <p style="font-size: 16px; color: #555; margin-top: 10px;">
                                            <strong>Recommendation:</strong> {advice}
                                        </p>
                                    </div>
                                    """,
                            unsafe_allow_html=True
                            )
                
                    except Exception as e:
                        st.error("⚠️ Prediction failed. Please check the input values or try again later.")
                        st.error(f"Error details: {e}")

        
# ............................................. Forecast Page ......................................................
        
        elif selection == "Forecast":
            st.title("📅 Air Quality Forecasting")
            st.write("Using Facebook Prophet, it forecast AQI for the selected duration.")
        
            data = load_data()
            if data.empty:
                st.warning("No data available for forecasting. Please upload historical data first.")
                st.stop()
            
            # Select forecast time
            forecast_hours = st.selectbox("Select Forecast Duration (hours)", [24, 48, 72], index=0)
            forecast_clicked = st.button("Generate Forecast", type="primary")
            if forecast_clicked:
               pmodel = train_prophet_model(data)
               with st.spinner("forecasting...."):
        
                  forecast_time = [datetime.now() + timedelta(hours=i) for i in range(1, forecast_hours + 1)]
                  # Get forecasted AQI data
                  forecast = forecast_aqi(pmodel, forecast_hours)
                  
                  forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_hours)
                  forecast_df.columns = ["Datetime", "Forecasted AQI", "Lower Bound", "Upper Bound"]
                  forecast_df["Datetime"] = forecast_time
    
            # Display forecast data
               st.write("### Forecasted AQI Data")
               st.dataframe(forecast_df)
               
               # Plot forecast
               plt.figure(figsize=(12, 6))
               plt.plot(forecast_df["Datetime"], forecast_df["Forecasted AQI"], label="Forecasted AQI", color='r')
               plt.fill_between(forecast_df["Datetime"], forecast_df["Lower Bound"], forecast_df["Upper Bound"], 
                                color='pink', alpha=0.3, label='Uncertainty Interval')
               plt.xlabel("Datetime")
               plt.ylabel("AQI")
               plt.title(f"{forecast_hours}-Hour AQI Forecast")
               plt.xticks(rotation=45)
               plt.grid(True)
               plt.legend()
               st.pyplot(plt)

            
           
# ................................................. About Page ...............................................

        st.sidebar.title("About")
        st.sidebar.info(
            """
            This web app predicts and forecasts Air Quality Index (AQI) of kathmandu using a pre-trained randomForest Regressor model and facebook prophet model resp.
            """
        )
        st.sidebar.markdown(" ")
        st.sidebar.markdown(
            """
            **Data Source:** [Kaggle](https://www.kaggle.com/sarojrawal)
            """
        )
        st.sidebar.markdown(
            """
            **Code:** [GitHub](https://github.com/saroj-2004/saroj)
            """
        )
        st.sidebar.header("⭐ Rate Us")
        rating = st.sidebar.slider("How would you rate our app?", 0, 10)
        st.sidebar.write(f"Your Rating: {rating}")
        
        if rating > 0:
            st.sidebar.success("Thank you for your feedback! 😊")
        
#................................................. End of App ...........................................
        
        
if __name__ == "__main__":
    main()

    