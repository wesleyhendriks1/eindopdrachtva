import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import ipywidgets as widgets
import seaborn as sns
import folium
from branca.colormap import linear
from folium.plugins import FeatureGroupSubGroup
from streamlit_folium import folium_static


# Set page config
st.set_page_config(
    page_title="Predicting Flight Delays",
    initial_sidebar_state="expanded"
)

# Title in sidebar
st.sidebar.markdown("<h1 style='color: #7F2704;'>Table of Contents</h1>", unsafe_allow_html=True)

# Rest of your Streamlit app code
selected_tab = st.sidebar.radio("Select Tab", ["Introduction", "Zurich Airport Analysis", "Prediction"], index=0)

if selected_tab == "Introduction":
    st.markdown("<h1 style='color: #7F2704;'>Predicting Flight Delays on Zurich Airport</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: #A63603;'>Introduction</h2>", unsafe_allow_html=True)
    st.write("""The idea of Case 3 was to create a model that predicts the delay for your next flight based on historical data. Now for this assignment, a lot of points could be improved on this and that is what has been done in this dashboard. We have addressed the points below to make this dashboard more accessible and attractive to end users: """)
    st.markdown("- The visualisations at Case 3 were all below each other, we have now chosen to have all visualisations in one overview. ")
    st.markdown("- The plot with the different Aircraft Types generated an error message when too many aircraft types were selected, we have now fixed this and have chosen to group it into 'Small', 'Narrow' and 'Wide' Aircraft. ")
    st.markdown("- The prediction model was very slow and included destination as the only factor. We have been able to ensure that the prediction is now many times faster, and includes three factors including: Inbound or Outbound, Origin/Destination Airport and Aircraft Type. ")

    st.markdown('''
    <style>
    [data-testid="stMarkdownContainer"] ul{
    list-style-position: inside;
    }
    </style>
    ''', unsafe_allow_html=True)

if selected_tab == "Zurich Airport Analysis":
    st.markdown("<h2 style='color: #7F2704;'>Zurich Airport Analysis</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns((2.0, 4.5 , 2.0), gap='large')

    with col1:
        st.markdown("<h3 style='color: #A63603;'>On Time/Delayed Flights</h3>", unsafe_allow_html=True)
        
        # Load data
        @st.cache_data
        def load_schedule_clean():
            scheduleclean = pd.read_csv('scheduleclean.csv')
            scheduleclean['STA_STD_ltc'] = pd.to_datetime(scheduleclean['STA_STD_ltc'])
            scheduleclean['ATA_ATD_ltc'] = pd.to_datetime(scheduleclean['ATA_ATD_ltc'])
            scheduleclean['Arrival_Status'] = scheduleclean['ATA_ATD_ltc'] - scheduleclean['STA_STD_ltc'] > pd.Timedelta(0)
            scheduleclean['Departure_Status'] = scheduleclean['STA_STD_ltc'] - scheduleclean['ATA_ATD_ltc'] > pd.Timedelta(0)
            return scheduleclean
    
        scheduleclean = load_schedule_clean()
    
        # Define colors
        delay_color = '#FDD0A2'
        ontime_color = '#F16913'
    
        # Calculate counts for the arrival plot
        total_arrival_flights = len(scheduleclean[scheduleclean['LSV'].str.contains('L')])
        arrival_delay_count = scheduleclean[scheduleclean['Arrival_Status'] & scheduleclean['LSV'].str.contains('L')].shape[0]
        arrival_ontime_count = total_arrival_flights - arrival_delay_count
        arrival_delay_percent = round(arrival_delay_count / total_arrival_flights * 100, 1)
        arrival_ontime_percent = round(arrival_ontime_count / total_arrival_flights * 100, 1)
    
        # Create DataFrame for the arrival plot
        data_arrival = {
            'Vlucht Status': ['Arrival Delay', 'On Time Arrival'],
            'Aantal Vluchten': [arrival_delay_count, arrival_ontime_count],
            'Percentage': [arrival_delay_percent, arrival_ontime_percent]
        }
        df_arrival = pd.DataFrame(data_arrival)
    
        # Calculate counts for the departure plot
        total_departure_flights = len(scheduleclean[scheduleclean['LSV'].str.contains('S')])
        departure_delay_count = scheduleclean[scheduleclean['Departure_Status'] & scheduleclean['LSV'].str.contains('S')].shape[0]
        departure_ontime_count = total_departure_flights - departure_delay_count
        departure_delay_percent = round(departure_delay_count / total_departure_flights * 100, 1)
        departure_ontime_percent = round(departure_ontime_count / total_departure_flights * 100, 1)
    
        # Create DataFrame for the departure plot
        data_departure = {
            'Vlucht Status': ['Departure Delay', 'On Time Departure'],
            'Aantal Vluchten': [departure_delay_count, departure_ontime_count],
            'Percentage': [departure_delay_percent, departure_ontime_percent]
        }
        df_departure = pd.DataFrame(data_departure)
    
        # Create plots
        arrival_fig = px.pie(df_arrival, values='Aantal Vluchten', names='Vlucht Status', hole=0.4,
                            color='Vlucht Status', color_discrete_map={'Arrival Delay': delay_color, 'On Time Arrival': ontime_color},
                            labels={'Aantal Vluchten': 'Aantal Vluchten'}, title='Total Arriving Flights')
    
        departure_fig = px.pie(df_departure, values='Aantal Vluchten', names='Vlucht Status', hole=0.4,
                                color='Vlucht Status', color_discrete_map={'Departure Delay': delay_color, 'On Time Departure': ontime_color},
                                labels={'Aantal Vluchten': 'Aantal Vluchten'}, title='Total Departing Flights')
    
        # Show arrival plot
        st.plotly_chart(arrival_fig, use_container_width=True)  # Adjust the width to fit the column
    
        # Show departure plot
        st.plotly_chart(departure_fig, use_container_width=True)  # Adjust the width to fit the column

############## 

    with col2:
        
        # Load aircraft types and max passenger capacity from Excel file
        aircraft_capacity = pd.read_excel('AC-MaxPassengers.xlsx')  # Update with your file path
        
        # Load schedule clean data
        scheduleclean = pd.read_csv('scheduleclean.csv')
        
        # Convert 'Delay' column to numeric format (hours)
        scheduleclean['Delay_hours'] = pd.to_timedelta(scheduleclean['Delay']).dt.total_seconds() / 3600
        
        # Grouping by Aircraft Type (ACT) and calculating the average Delay
        avg_delay_per_aircraft_type = scheduleclean.groupby('ACT')['Delay_hours'].mean().reset_index()
        
        # Renaming the columns for clarity
        avg_delay_per_aircraft_type.columns = ['ACT', 'Average_Delay_hours']
        
        # Group aircraft types based on max passenger capacity
        grouped_aircraft = aircraft_capacity.groupby('Max passengers')
        
        # Create dropdown menu options
        dropdown_options = ['All Small Aircraft', 'All Narrow Aircraft', 'All Wide Aircraft']
        
        # Define a function to plot delay per aircraft category
        def plot_delay_per_aircraft_category(category):
            if category.startswith('All'):
                # Plot all aircraft types
                if category == 'All Small Aircraft':
                    filtered_data = avg_delay_per_aircraft_type[avg_delay_per_aircraft_type['ACT'].isin(aircraft_capacity[aircraft_capacity['Max passengers'] <= 100]['Aircraft'])]
                elif category == 'All Narrow Aircraft':
                    filtered_data = avg_delay_per_aircraft_type[avg_delay_per_aircraft_type['ACT'].isin(aircraft_capacity[(aircraft_capacity['Max passengers'] > 100) & (aircraft_capacity['Max passengers'] <= 250)]['Aircraft'])]
                elif category == 'All Wide Aircraft':
                    filtered_data = avg_delay_per_aircraft_type[avg_delay_per_aircraft_type['ACT'].isin(aircraft_capacity[aircraft_capacity['Max passengers'] > 250]['Aircraft'])]
            else:
                # Plot specific aircraft type
                filtered_data = avg_delay_per_aircraft_type[avg_delay_per_aircraft_type['ACT'].isin(grouped_aircraft.get_group(category)['Aircraft'])]
            
            # Create a bar plot
            fig = px.bar(filtered_data, x='ACT', y='Average_Delay_hours',
                     title='Average Delay per Aircraft Type',
                     labels={'ACT': 'Aircraft Type', 'Average_Delay_hours': 'Average Delay (hours)'},
                     width=800, height=500, 
                     color='Average_Delay_hours', 
                     color_continuous_scale='Oranges')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig)
        
#################
        
        
        # Define airport code to airport name mapping
        icao_to_airport = {
            'EDDT': 'Berlin Tegel',
            'EGLL': 'London Heathrow',
            'EHAM': 'Amsterdam Airport Schiphol',
            'EDDF': 'Frankfurt Main',
            'LOWW': 'Vienna Airport',
            'LEMD': 'Madrid-Barajas Airport',
            'EGLC': 'London City',
            'EDDH': 'Hamburg Airport',
            'EDDL': 'Dusseldorf Airport',
            'LFPG': 'Paris Charles de Gaulle',
            'LSGG': 'Geneva Airport',
            'LPPT': 'Lisbon Airport'
        }
        
        # Load data
        def load_flight_delays():
            return pd.read_csv('flightdelays.csv')
        
        # Process data based on flight type
        def process_data(flightdelays, flight_type):
            if flight_type == 'Inbound':
                filtered_df = flightdelays[flightdelays['LSV'] == 'L']
            else:
                filtered_df = flightdelays[flightdelays['LSV'] == 'S']
                
            top_destinations = filtered_df['Org/Des'].value_counts().reset_index().head(10)
            top_destinations.columns = ['Destination', 'Count']
            
            # Replace ICAO codes with airport names
            top_destinations['Destination'] = top_destinations['Destination'].map(icao_to_airport)
            
            # Add ranking
            top_destinations.index = top_destinations.index + 1
            top_destinations.index.name = 'Rank'
            
            return top_destinations
        
        # Main code
        st.markdown("<h3 style='color: #A63603;'>Top 10 Destinations with Most Delay Counts</h3>", unsafe_allow_html=True)
        flight_type = st.selectbox('Select Inbound or Outbound Flight', ['Inbound', 'Outbound'])
        
        # Load flight delays data
        flightdelays = load_flight_delays()
        
        # Process data based on selected flight type
        top_destinations = process_data(flightdelays, flight_type)
        
        # Load your dataset (assuming 'test2' is your DataFrame)
        test2 = pd.read_csv('test2.csv')
        
        # Convert latitude and longitude columns to float
        test2['Latitude'] = test2['Latitude'].astype(float)
        test2['Longitude'] = test2['Longitude'].astype(float)
        
        # Assuming 'LSV' column contains the status of the flight: 'L' for inbound, 'S' for outbound
        
        # Count the frequency of each location for inbound flights
        inbound_counts = test2[test2['LSV'] == 'L'].groupby(['Latitude', 'Longitude']).size().reset_index(name='Count')
        
        # Count the frequency of each location for outbound flights
        outbound_counts = test2[test2['LSV'] == 'S'].groupby(['Latitude', 'Longitude']).size().reset_index(name='Count')
        
        # Define a color gradient for the circle markers using shades of orange
        colormap = linear.Oranges_09.scale(min(inbound_counts['Count'].min(), outbound_counts['Count'].min()),
                                            max(inbound_counts['Count'].max(), outbound_counts['Count'].max()))
        
        # Create a map centered on Europe with a zoom level that shows most of Europe
        map_europe = folium.Map(location=[48.5260, 11.2551], zoom_start=3.5, tiles='cartodbpositron', width='100%', height='400px')
        
        # Add a marker for Zurich Airport
        zurich_airport_coords = [47.4647, 8.5492]
        folium.Marker(location=zurich_airport_coords, popup='Zurich Airport').add_to(map_europe)
        
        # Add gradient circle markers based on selected flight type
        if flight_type == 'Inbound':
            data = inbound_counts
        else:
            data = outbound_counts
        
        for index, row in data.iterrows():
            coords = [row['Latitude'], row['Longitude']]
            count = row['Count']
            # Map the frequency to a color from the gradient
            color = colormap(count)
            folium.Circle(location=coords, radius=50000, color=color, fill=True, fill_color=color).add_to(map_europe)
        
        # Add the color scale to the map
        colormap.add_to(map_europe)
        
        # Display the map using Streamlit's folium_static
        folium_static(map_europe)


        st.markdown("<h3 style='color: #A63603;'>Average Delay per Aircraft Type</h3>", unsafe_allow_html=True)

        # Create a dropdown menu widget
        dropdown_menu = st.selectbox(
            "Aircraft Category:",
            dropdown_options,
            index=0
        )
        
        # Display the dropdown menu and plot the initial graph
        plot_delay_per_aircraft_category(dropdown_menu)
    

    with col3:

        st.markdown("<h3 style='color: #A63603;'>Top 10 Destinations with Most Delay Counts</h3>", unsafe_allow_html=True)
       
        # Display the table
        st.table(top_destinations.style.set_properties(**{'text-align': 'center'}).format({'Count': '{:,}'}))

#################################

elif selected_tab == "Prediction":
    st.markdown("<h2 style='color: #7F2704;'>Predicting Flight Delays</h2>", unsafe_allow_html=True)
    
    def main():

        st.write('Based on the top 10 destinations with the most delays, this model calculates the most likely delay you will have if you fly to or from one of these destinations. This is based on several historical factors: Inbound (L) or Outbound (S) flight, Destination (ICAO code) and Aircraft Type.')
        
        # Load dataset
        flight_data = pd.read_csv('prediction.csv')  # Update with your dataset name
    
        # Step 1: Filter dataset based on user input for LSV
        lsv_value = st.text_input("Enter L for Inbound and S for Outbound: ")
        filtered_data = flight_data[flight_data['LSV'] == lsv_value]
    
        # Step 2: Allow user to input ICAO code (Org/Des)
        icao_code = st.text_input("Enter the ICAO code for the destination, e.g. 'EHAM' for Amsterdam Airport Schiphol).")
        
        if icao_code:
            filtered_data = filtered_data[filtered_data['Org/Des'] == icao_code]
        
            # Check if ICAO code exists in the filtered dataset
            if filtered_data.empty:
                st.error("ICAO code not found.")
                return
    
        # Step 3: Allow user to input aircraft type (ACT)
        aircraft_type = st.text_input("Enter the aircraft type: e.g., A319 for Airbus A319.")
        
        if aircraft_type:
            filtered_data = filtered_data[filtered_data['ACT'] == aircraft_type]
    
            # Check if aircraft type exists in the filtered dataset
            if filtered_data.empty:
                st.error("Aircraft Type not found.")
                return
    
        if st.button('Calculate'):
            # Features for prediction: 'Org/Des', 'ACT'
            X = filtered_data[['Org/Des', 'ACT']]
            y = filtered_data['Delay_Hours']
        
            # Step 4: Split the Data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
            # Step 5: Feature Encoding (One-Hot Encoding for categorical features 'Org/Des' and 'Aircraft_Type')
            encoder = OneHotEncoder(handle_unknown='ignore')
            X_train_encoded = encoder.fit_transform(X_train)
            X_test_encoded = encoder.transform(X_test)
        
            # Step 6: Model Selection and Training (using Random Forest)
            model = RandomForestRegressor()  # Initialize Random Forest regressor
            model.fit(X_train_encoded, y_train)
        
            # Step 7: Model Evaluation
            predictions = model.predict(X_test_encoded)
            mae = mean_absolute_error(y_test, predictions)
        
            # Step 8: Prediction
            predicted_delay = model.predict(X_test_encoded)
        
            # Step 9: Display the predicted delay with appropriate color
            predicted_delay_minutes = predicted_delay * 60
            
            # Define color based on the value of predicted delay
            color = 'red' if predicted_delay_minutes[0] > 0 else 'green'
            
            # Display the predicted delay using custom HTML/CSS for text color
            st.markdown(f"<p style='color:{color};'>Predicted Delay (minutes): {predicted_delay_minutes[0]}</p>", unsafe_allow_html=True)





    if __name__ == "__main__":
        main()










    


