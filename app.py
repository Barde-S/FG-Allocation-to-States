import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import numpy as np
import openpyxl
regions = {
    'North Central': ['Benue', 'Kogi', 'Kwara', 'Nasarawa', 'Niger', 'Plateau', 'Federal Capital Territory'],
    'North East': ['Adamawa', 'Bauchi', 'Borno', 'Gombe', 'Taraba', 'Yobe'],
    'North West': ['Jigawa', 'Kaduna', 'Kano', 'Katsina', 'Kebbi', 'Sokoto', 'Zamfara'],
    'South East': ['Abia', 'Anambra', 'Ebonyi', 'Enugu', 'Imo'],
    'South South': ['Akwa Ibom', 'Bayelsa', 'Cross River', 'Delta', 'Edo', 'Rivers'],
    'South West': ['Ekiti', 'Lagos', 'Ogun', 'Ondo', 'Osun', 'Oyo']
}
# Load data
df = pd.read_excel('FAAC DATA - Data Community.xlsx', sheet_name='State')
df = df.iloc[:, :-7]
lgas = pd.read_excel('FAAC DATA - Data Community.xlsx', sheet_name='LGA')
for col in lgas.columns:
    if lgas[col].isna().sum() == 776:
        lgas = lgas.drop(columns=col) # Remove inplace=True and assignment
for col in lgas.columns[2:]:
    if lgas[col].isna().sum() > 0:
        lgas[col] = pd.to_numeric(lgas[col], errors='coerce')
        lgas[col].fillna(np.mean(lgas[col]), inplace=True)
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
lgas['STATE'] = lgas.STATE.str.capitalize()
# Reverse mapping for convenience
state_to_region = {state: region for region, states in regions.items() for state in states}
# Add a 'Region' column to the DataFrame
df['Region'] = df['State'].map(state_to_region)
df_melted = df.melt(id_vars=['State', 'Region'], var_name='Date', value_name='Allocation')
# Convert Date column to datetime format
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format='%b-%Y')
# Create a pivot table to calculate average allocations by region for each month
pivot_table_avg = pd.pivot_table(
    df_melted,
    values='Allocation',
    index=df_melted['Date'].dt.strftime('%b-%Y'),
    columns='Region',
    aggfunc='mean'
)
pivot_table_sum = pd.pivot_table(
    df_melted,
    values='Allocation',
    index=df_melted['Date'].dt.strftime('%b-%Y'),
    columns='Region',
    aggfunc='sum'
)
# Reverse mapping for convenience
state_to_region = {STATE: region for region, states in regions.items() for STATE in states}
# Add a 'Region' column to the DataFrame
lgas['Region'] = lgas['STATE'].map(state_to_region)
lgas.head()
# Streamlit app structure
with st.sidebar:
    selected = option_menu(
        menu_title="Explore",
        options=["Static", "Dynamic"],
        menu_icon="cast"
    )
if selected == "Static":
    st.header("                        FAAC Allocation")
   
    numeric_columns = df.select_dtypes(include='number')
    total_allocations_by_state = numeric_columns.set_index(df['State']).sum(axis=1).sort_values(ascending=False)
    total_allocations_by_state = total_allocations_by_state.reset_index()
    total_allocations_by_state.columns = ['State', 'Allocation']
    
    # Set the style
    #sns.set_style("whitegrid")
    
    fig = px.bar(
    total_allocations_by_state,
    x='State',
    y='Allocation',
    color='State',  # Use color differentiation for each bar
    title='Total Allocations by State (2007-2024)',
    labels={'State': 'State', 'Total Allocation': 'Allocation'}
    #text='Total Allocation'  # Show values on the bars
)

# Customize the layout for better display
    fig.update_layout(
    title={'font': {'size': 30, 'color': 'white', 'family': 'Arial'}},
    xaxis_title='State',
    yaxis_title='Total Allocation',
    xaxis_tickangle=-90,  # Rotate x-axis labels for better readability
    xaxis={'tickfont': {'size': 12}},
    yaxis={'tickfont': {'size': 12}},
    template='plotly_white'
)

# Display the plot in Streamlit
    st.plotly_chart(fig)
    
    
    regions = {'North Central': ['Benue', 'Kogi', 'Kwara', 'Nasarawa', 'Niger', 'Plateau', 'Federal Capital Territory'],
              'North East': ['Adamawa', 'Bauchi', 'Borno', 'Gombe', 'Taraba', 'Yobe'],
              'North West': ['Jigawa', 'Kaduna', 'Kano', 'Katsina', 'Kebbi', 'Sokoto', 'Zamfara'],
              'South East': ['Abia', 'Anambra', 'Ebonyi', 'Enugu', 'Imo'],
              'South South': ['Akwa Ibom', 'Bayelsa', 'Cross River', 'Delta', 'Edo', 'Rivers'],
              'South West': ['Ekiti', 'Lagos', 'Ogun', 'Ondo', 'Osun', 'Oyo']
    }
    # Reverse mapping for convenience
    state_to_region = {state: region for region, states in regions.items() for state in states}
    # Add a 'Region' column to the DataFrame
    df['Region'] = df['State'].map(state_to_region)
    df_melted = df.melt(id_vars=['State', 'Region'], var_name='Date', value_name='Allocation')
    # Convert Date column to datetime format
    df_melted['Date'] = pd.to_datetime(df_melted['Date'], format='%b-%Y')
    # Create a pivot table to calculate average allocations by region for each month
    pivot_table_avg = pd.pivot_table(
        df_melted,
        values='Allocation',
        index=df_melted['Date'].dt.strftime('%b-%Y'),
        columns='Region',
        aggfunc='mean'
    )
    pivot_table_sum = pd.pivot_table(
    df_melted,
    values='Allocation',
    index=df_melted['Date'].dt.strftime('%b-%Y'),
    columns='Region',
    aggfunc='sum'
    )


       # Define the correct order for months
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # Convert index to categorical with specified order
    pivot_table_avg.index = pd.Categorical(pivot_table_avg.index, categories=month_order, ordered=True)
    monthly_avg_pivot = pivot_table_avg.sort_index()
    # Plot monthly average allocations by region
    # plt.figure()
    # for region in monthly_avg_pivot.columns:
    #     plt.plot(monthly_avg_pivot.index, monthly_avg_pivot[region], marker='o', label=region)
    # plt.title('Monthly Average Allocations by Region')
    # plt.xlabel('Month')
    # plt.ylabel('Average Allocation')
    # plt.legend()
    # plt.grid(True)
    # st.pyplot(plt)
    # st.subheader('Inflation Trends')



# Melt the DataFrame to long format for easier manipulation
    df_melted = df.melt(id_vars=['State', 'Region'], var_name='Date', value_name='Allocation')

# Convert Date column to datetime format
    df_melted['Date'] = pd.to_datetime(df_melted['Date'], format='%b-%Y', errors='coerce')

# Drop rows with invalid dates
    df_melted.dropna(subset=['Date'], inplace=True)

# Create a pivot table to calculate average allocations by region for each month
    pivot_table_avg = pd.pivot_table(
    df_melted,
    values='Allocation',
    index=df_melted['Date'].dt.year,
    columns='Region',
    aggfunc='mean'
)

    pivot_table_sum = pd.pivot_table(
    df_melted,
    values='Allocation',
    index=df_melted['Date'].dt.year,
    columns='Region',
    aggfunc='sum'
)

# Ensure the transposed pivot tables are properly indexed for plotting
    pivot_table_sum_transposed = pivot_table_sum
    pivot_table_avg_transposed = pivot_table_avg


#     for region in pivot_table_sum_transposed.columns:
#         plt.plot(pivot_table_sum_transposed[region], label=region)

#     plt.title('Sum Allocations Time Series by Region')
#     plt.xlabel('Date')
#     plt.ylabel('Sum Allocation')
#     plt.legend()
#     plt.grid(True)

# # Display the plot in Streamlit
#     st.pyplot(plt)





    
    fig_avg = go.Figure()

    for region in pivot_table_sum_transposed.columns:
        fig_avg.add_trace(go.Scatter(
        x=pivot_table_sum_transposed.index,
        y=pivot_table_sum_transposed[region],
        mode='lines',
        name=region
    ))

    fig_avg.update_layout(
   # title='Yearly Total Allocations Time Series by Region',
    title='Yearly Total Allocations Time Series by Region',
    xaxis_title='Date',
    yaxis_title='Total Allocation',
    legend_title='Region',
    template='plotly_white'
)

    st.plotly_chart(fig_avg)
    
    
        
    
# Plotting average allocations time series for each region using Plotly
    fig_avg = go.Figure()

    for region in pivot_table_avg_transposed.columns:
        fig_avg.add_trace(go.Scatter(
        x=pivot_table_avg_transposed.index,
        y=pivot_table_avg_transposed[region],
        mode='lines',
        name=region
    ))

    fig_avg.update_layout(
    title='Yearly Average Allocations Time Series by Region',
    xaxis_title='Date',
    yaxis_title='Average Allocation',
    legend_title='Region',
    template='plotly_white'
)

    st.plotly_chart(fig_avg)

# Financial and Monetary Indicators
    st.subheader('')
    st.write("""
""")

# Group by region and sum allocations
    region_totals = df_melted.groupby('Region')['Allocation'].sum().reset_index()

# Create a pie chart using Plotly
    fig1 = px.pie(region_totals, values='Allocation', names='Region',
              title='Total Allocations Share by Region')

# Display the pie chart
    st.plotly_chart(fig1)

    st.subheader('')
    st.write("""
""")

# Find the top states by region
    top_states_by_region = df_melted.loc[df_melted.groupby('Region')['Allocation'].idxmax()]

# Sort the result by Allocation
    top_states_by_region = top_states_by_region.sort_values(by='Allocation', ascending=False)

# Create the bar plot using Plotly
    fig2 = px.bar(
    top_states_by_region,
    x='State',
    y='Allocation',
    color='Region',
    labels={'Allocation': 'Total Allocation'},
    title='Top States with Most Allocation from Each Region',
    template='plotly_white',
    color_discrete_sequence=px.colors.qualitative.Plotly
)

# Customize the layout for better readability
    fig2.update_layout(
    autosize=False,
    width=1850,
    height=600,
    xaxis_title='State',
    yaxis_title='Total Allocation',
    xaxis_tickangle=-90,
    title_font=dict(size=20, family='Arial', color='White'),
    xaxis=dict(tickfont=dict(size=10)),
    yaxis=dict(tickfont=dict(size=10)),
    legend_title_text='Region'
)

# Show the bar plot
    st.plotly_chart(fig2)


    # Calculate the total allocations by LGC, excluding the 'Region' column
    # total_allocations_by_state = lgas.set_index(['LGC']).drop('Region', axis=1).apply(pd.to_numeric, errors='coerce').sum(axis=1).sort_values(ascending=False).head(10)

    # # Convert the Series to a DataFrame for easier plotting with Plotly
    # total_allocations_by_state_df = total_allocations_by_state.reset_index()
    # # total_allocations_by_state_df.columns = ['LGC', 'Total Allocation']
    # total_allocations_by_state_df['y'] =  ((total_allocations_by_state_df['Total Allocation'])/1000)
    # # (total_allocations_by_state_df.y)/1000

    
#     # Create the bar chart using Plotly
#     fig = px.bar(
#     total_allocations_by_state_df,
#     x='LGC',
#     y='y',
#     title='Top Ten (10) LGC with Most Total Allocations',
#     labels={'LGC': 'LGC', 'Total Allocation': 'Total Allocation'}
#         )

#     # Customize the layout for better display
#     fig.update_layout(
#     xaxis_title='LGC',
#     yaxis_title='Total Allocation',
#     xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
#     template='plotly_white'
# )

#     # Display the plot in Streamlit
#     st.plotly_chart(fig)


    melted = lgas.melt(id_vars=['LGC', 'Region'], var_name='Date', value_name='Allocation')

# Convert Date column to datetime format
    melted['Date'] = pd.to_datetime(melted['Date'], format='%b-%Y', errors='coerce')

# Drop rows with invalid dates
    melted.dropna(subset=['Date'], inplace=True)

# Ensure that 'Allocation' column contains numeric values
    melted['Allocation'] = pd.to_numeric(melted['Allocation'], errors='coerce')

# Drop rows with invalid allocation values
    melted.dropna(subset=['Allocation'], inplace=True)

# Create a pivot table to calculate sum allocations by LGC for each year
    pivot_table_sum = pd.pivot_table(
    melted,
    values='Allocation',
    index=melted['Date'].dt.year,
    columns='LGC',
    aggfunc='sum'
)

# Find the top 10 LGCs by total allocation
    top_10_lgcs = melted.groupby('LGC')['Allocation'].sum().sort_values(ascending=False).head(10).index

# Filter the melted DataFrame to include only the top 10 LGCs
    top_10_melted = melted[melted['LGC'].isin(top_10_lgcs)]

# Create the bar chart using Plotly
    fig = px.bar(
    top_10_melted.sort_values(by='Allocation', ascending=False),
    x='LGC',
    y='Allocation',
    color='LGC',
    title='Top Ten (10) LGC with Most Total Allocations',
    labels={'LGC': 'LGC', 'Total Allocation': 'Allocation'}
)

# Customize the layout for better display
    fig.update_layout(
    xaxis_title='LGC',
    yaxis_title='Total Allocation',
    xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
    template='plotly_white'
)

# Display the plot in Streamlit
    st.plotly_chart(fig)


        

if selected == "Dynamic":
    st.header("FAAC Allocation")
    st.subheader('')
    st.write("""
    """)
    numeric_columns = df.select_dtypes(include='number')
    total_allocations_by_state = numeric_columns.set_index(df['State']).sum(axis=1).sort_values(ascending=False)



# Define the order of months for proper plotting
    month_order = ['2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06',
               '2023-07', '2023-08', '2023-09', '2023-10', '2023-11', '2023-12']

# Convert Date column to datetime format
    df_melted['Date'] = pd.to_datetime(df_melted['Date'], format='%Y-%m')

# Extract the year and month from the Date column
    df_melted['Year'] = df_melted['Date'].dt.year
    df_melted['Month'] = df_melted['Date'].dt.strftime('%Y-%m')

    allocations_by_year = df_melted

# Get unique states and years from the DataFrame
    unique_states = [lgc.capitalize() for lgc in allocations_by_year['State'].unique()]
    unique_years = allocations_by_year['Year'].unique()

    st.title("State Allocations Analysis")

# Prompt user to select the first state
    state_selected_a = st.selectbox("Select the first State:", unique_states, key='state_e')

# Prompt user to select the year range
    start_year, end_year = st.slider("Select a year range:", min(unique_years), max(unique_years), (min(unique_years), max(unique_years)), key='range_slider1')

# Prompt user to select whether they want the total sum or average
    plot_type = st.radio("Select the type of plot:", ("Total Sum", "Average"), key='plot_type_radio1')

    if start_year == end_year:
    # Filter data by month within the selected year for the state
        filtered_data_a = allocations_by_year[
        (allocations_by_year['State'] == state_selected_a) & 
        (allocations_by_year['Year'] == start_year)
    ]
    
        if plot_type == "Total Sum":
        # Create monthly line plot for total sum
            summed_data_a = filtered_data_a.groupby('Month')['Allocation'].sum().reindex(month_order).fillna(0).reset_index()
            fig = px.line(summed_data_a, x='Month', y='Allocation', markers=True, title=f'Total Allocations by Month for {state_selected_a} in {start_year}')
        else:
        # Create monthly line plot for average
            avg_data_a = filtered_data_a.groupby('Month')['Allocation'].mean().reindex(month_order).fillna(0).reset_index()
            fig = px.line(avg_data_a, x='Month', y='Allocation', markers=True, title=f'Average Allocations by Month for {state_selected_a} in {start_year}')
    else:
    # Filter data based on user selection for yearly plot
        filtered_data_a = allocations_by_year[
        (allocations_by_year['State'] == state_selected_a) &
        (allocations_by_year['Year'] >= start_year) &
        (allocations_by_year['Year'] <= end_year)
    ]
    
        if plot_type == "Total Sum":
        # Create yearly line plot for total sum
            summed_data_a = filtered_data_a.groupby('Year')['Allocation'].sum().reset_index()
            fig = px.line(summed_data_a, x='Year', y='Allocation', markers=True, title=f'Total Allocations for {state_selected_a} Over the Years')
        else:
        # Create yearly line plot for average
            avg_data_a = filtered_data_a.groupby('Year')['Allocation'].mean().reset_index()
            fig = px.line(avg_data_a, x='Year', y='Allocation', markers=True, title=f'Average Allocations for {state_selected_a} Over the Years')

    fig.update_layout(xaxis_title='Year' if start_year != end_year else 'Month', yaxis_title='Total Allocation')
    fig.update_xaxes(type='category' if start_year == end_year else 'linear', categoryorder='array' if start_year == end_year else None, categoryarray=month_order if start_year == end_year else None)
    st.plotly_chart(fig)
   


    month_order = ['2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06',
                   '2023-07', '2023-08', '2023-09', '2023-10', '2023-11', '2023-12']

# Melt the DataFrame to long format for easier manipulation
    df_melted = df.melt(id_vars=['State', 'Region'], var_name='Date', value_name='Allocation')

# Convert Date column to datetime format
    df_melted['Date'] = pd.to_datetime(df_melted['Date'], format='%Y-%m')

# Extract the year and month from the Date column
    df_melted['Year'] = df_melted['Date'].dt.year
    df_melted['Month'] = df_melted['Date'].dt.strftime('%Y-%m')

    allocations_by_year = df_melted

    # Get unique states and years from the DataFrame
    unique_states = [lgc.capitalize() for lgc in allocations_by_year['State'].unique()]
    unique_years = allocations_by_year['Year'].unique()

    # Streamlit interface
    st.title("State Allocations Analysis")

    # Prompt user to select the first state
    state_selected_a = st.selectbox("Select the first State:", unique_states, key='state_a')

    # Prompt user to select the second state
    state_selected_b = st.selectbox("Select the second State:", unique_states, key='state_b')

    # Prompt user to select the year range
    start_year, end_year = st.slider("Select a year range:", min(unique_years), max(unique_years), (min(unique_years), max(unique_years)), key='range_slider2')

    # Prompt user to select whether they want the total sum or average
    plot_type = st.radio("Select the type of plot:", ("Total Sum", "Average"), key='plot_type_radio2')

    # Check if start year and end year are the same
    if start_year == end_year:
    # Filter data by month within the selected year for both states
        filtered_data_a = allocations_by_year[
        (allocations_by_year['State'] == state_selected_a) & 
        (allocations_by_year['Year'] == start_year)
    ]
        filtered_data_b = allocations_by_year[
        (allocations_by_year['State'] == state_selected_b) & 
        (allocations_by_year['Year'] == start_year)
    ]
        if plot_type == "Total Sum":
        # Create monthly line plot for total sum
            summed_data_a = filtered_data_a.groupby('Month')['Allocation'].sum().reindex(month_order).fillna(0)
            summed_data_b = filtered_data_b.groupby('Month')['Allocation'].sum().reindex(month_order).fillna(0)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=summed_data_a.index, y=summed_data_a.values, mode='lines+markers', name=f"{state_selected_a} - Total Sum"))
            fig.add_trace(go.Scatter(x=summed_data_b.index, y=summed_data_b.values, mode='lines+markers', name=f"{state_selected_b} - Total Sum"))
            fig.update_layout(title=f'Total Allocations by Month for {state_selected_a} and {state_selected_b} in {start_year}', xaxis_title='Month', yaxis_title='Total Allocation')
        else:
            # Create monthly line plot for average
            avg_data_a = filtered_data_a.groupby('Month')['Allocation'].mean().reindex(month_order).fillna(0)
            avg_data_b = filtered_data_b.groupby('Month')['Allocation'].mean().reindex(month_order).fillna(0)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=avg_data_a.index, y=avg_data_a.values, mode='lines+markers', name=f"{state_selected_a} - Average"))
            fig.add_trace(go.Scatter(x=avg_data_b.index, y=avg_data_b.values, mode='lines+markers', name=f"{state_selected_b} - Average"))
            fig.update_layout(title=f'Average Allocations by Month for {state_selected_a} and {state_selected_b} in {start_year}', xaxis_title='Month', yaxis_title='Average Allocation')
    else:
        filtered_data_a = allocations_by_year[
        (allocations_by_year['State'] == state_selected_a) &
        (allocations_by_year['Year'] >= start_year) &
        (allocations_by_year['Year'] <= end_year)
    ]
        filtered_data_b = allocations_by_year[
        (allocations_by_year['State'] == state_selected_b) &
        (allocations_by_year['Year'] >= start_year) &
        (allocations_by_year['Year'] <= end_year)
    ]
        if plot_type == "Total Sum":
        # Create yearly line plot for total sum
            summed_data_a = filtered_data_a.groupby('Year')['Allocation'].sum().reset_index()
            summed_data_b = filtered_data_b.groupby('Year')['Allocation'].sum().reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=summed_data_a['Year'], y=summed_data_a['Allocation'], mode='lines+markers', name=f"{state_selected_a} - Total Sum"))
            fig.add_trace(go.Scatter(x=summed_data_b['Year'], y=summed_data_b['Allocation'], mode='lines+markers', name=f"{state_selected_b} - Total Sum"))
            fig.update_layout(title=f'Comparison of Total Allocations between {state_selected_a} and {state_selected_b} Over the Years', xaxis_title='Year', yaxis_title='Total Allocation')
        else:
            avg_data_a = filtered_data_a.groupby('Year')['Allocation'].mean().reset_index()
            avg_data_b = filtered_data_b.groupby('Year')['Allocation'].mean().reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=avg_data_a['Year'], y=avg_data_a['Allocation'], mode='lines+markers', name=f"{state_selected_a} - Average"))
            fig.add_trace(go.Scatter(x=avg_data_b['Year'], y=avg_data_b['Allocation'], mode='lines+markers', name=f"{state_selected_b} - Average"))
            fig.update_layout(title='Average Allocations by State Over the Years', xaxis_title='Year', yaxis_title='Average Allocation')

    # Display the plot
    st.plotly_chart(fig)




    # Define the order of months for proper plotting
    month_order = ['2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06',
                   '2023-07', '2023-08', '2023-09', '2023-10', '2023-11', '2023-12']

    # Melt the DataFrame to long format for easier manipulation
    df_melted = lgas.melt(id_vars=['STATE', 'LGC', 'Region'], var_name='Date', value_name='Allocation')

    # Convert Date column to datetime format
    df_melted['Date'] = pd.to_datetime(df_melted['Date'], format='%b-%Y')

    # Extract the year and month from the Date column
    df_melted['Year'] = df_melted['Date'].dt.year
    df_melted['Month'] = df_melted['Date'].dt.strftime('%Y-%m')

    # Now `df_melted` is the DataFrame we'll use
    allocations_by_year = df_melted

    # Get unique states and years from the DataFrame
    unique_states = [lgc for lgc in allocations_by_year['LGC'].unique()]
    unique_years = allocations_by_year['Year'].unique()

    # Streamlit App
    st.title("Allocation to LGC Analysis")


    state_selected_a = st.selectbox("Select LGC:", unique_states, key='state_c')

# Select the year range
    start_year, end_year = st.select_slider(
    "Select the year range:",
    options=unique_years,
    value=(min(unique_years), max(unique_years))
)

# Select the type of plot
    plot_type = st.radio(
    "Select the type of plot:",
    ('Total Sum', 'Average'),
    key='plot_type_radio3'
)

    # Filter data based on user selection
    filtered_data_a = allocations_by_year[
    (allocations_by_year['LGC'] == state_selected_a) &
    (allocations_by_year['Year'] >= start_year) &
    (allocations_by_year['Year'] <= end_year)]

    fig = go.Figure()

    if start_year == end_year:
        if plot_type == 'Total Sum':
        # Create monthly line plot for total sum
            summed_data_a = filtered_data_a.groupby('Month')['Allocation'].sum().reindex(month_order).fillna(0)
        
            fig.add_trace(go.Scatter(x=summed_data_a.index, y=summed_data_a.values, mode='lines+markers', name=f"{state_selected_a} - Total Sum"))
           
            fig.update_layout(title=f'Total Allocations by Month for {state_selected_a} in {start_year}')
        else:
        # Create monthly line plot for average
            avg_data_a = filtered_data_a.groupby('Month')['Allocation'].mean().reindex(month_order).fillna(0)
           
            fig.add_trace(go.Scatter(x=avg_data_a.index, y=avg_data_a.values, mode='lines+markers', name=f"{state_selected_a} - Average"))
          
            fig.update_layout(title=f'Average Allocations by Month for {state_selected_a} in {start_year}')
    else:
        if plot_type == 'Total Sum':
        # Create yearly line plot for total sum
            summed_data_a = filtered_data_a.groupby('Year')['Allocation'].sum().reset_index()
          
            fig.add_trace(go.Scatter(x=summed_data_a['Year'], y=summed_data_a['Allocation'], mode='lines+markers', name=f"{state_selected_a} - Total Sum"))
           
            fig.update_layout(title=f'Total Allocations to {state_selected_a} Over the Years')
        else:
        # Create yearly line plot for average
            avg_data_a = filtered_data_a.groupby('Year')['Allocation'].mean().reset_index()
           
            fig.add_trace(go.Scatter(x=avg_data_a['Year'], y=avg_data_a['Allocation'], mode='lines+markers', name=f"{state_selected_a} - Average"))
           
            fig.update_layout(title=f'Average Allocations to {state_selected_a} Over the Years')

    fig.update_xaxes(title_text='Year' if start_year != end_year else 'Month')
    fig.update_yaxes(title_text='Total Allocation')
    fig.update_layout(legend=dict(x=0, y=1, traceorder='normal'), xaxis=dict(tickangle=90))

    # Display the plot
    st.plotly_chart(fig)







    

    # Select the first state
    state_selected_a = st.selectbox("Select the first LGC:", unique_states, key='state_g')

# Select the year range
    start_year, end_year = st.select_slider(
    "Select the year range:",
    key='date_g',
    options=unique_years,
    value=(min(unique_years), max(unique_years))
)

# Select the type of plot
    plot_type = st.radio(
    "Select the type of plot:",
    ('Total Sum', 'Average'),
    key='plot_type_radiog'
)

    # Filter data based on user selection
    filtered_data_a = allocations_by_year[
    (allocations_by_year['LGC'] == state_selected_a) &
    (allocations_by_year['Year'] >= start_year) &
    (allocations_by_year['Year'] <= end_year)
]

    # Filter data based on user selection for second LGC
    state_selected_b = st.selectbox("Select the second LGC:", unique_states, key='states_g')

    filtered_data_b = allocations_by_year[
    (allocations_by_year['LGC'] == state_selected_b) &
    (allocations_by_year['Year'] >= start_year) &
    (allocations_by_year['Year'] <= end_year)
]

    fig = go.Figure()

    if start_year == end_year:
        if plot_type == 'Total Sum':
        # Create monthly line plot for total sum
            summed_data_a = filtered_data_a.groupby('Month')['Allocation'].sum().reindex(month_order).fillna(0)
            summed_data_b = filtered_data_b.groupby('Month')['Allocation'].sum().reindex(month_order).fillna(0)
        
            fig.add_trace(go.Scatter(x=summed_data_a.index, y=summed_data_a.values, mode='lines+markers', name=f"{state_selected_a} - Total Sum"))
            fig.add_trace(go.Scatter(x=summed_data_b.index, y=summed_data_b.values, mode='lines+markers', name=f"{state_selected_b} - Total Sum"))
        
            fig.update_layout(title=f'Total Allocations by Month for {state_selected_a} and {state_selected_b} in {start_year}')
        else:
        # Create monthly line plot for average
            avg_data_a = filtered_data_a.groupby('Month')['Allocation'].mean().reindex(month_order).fillna(0)
            avg_data_b = filtered_data_b.groupby('Month')['Allocation'].mean().reindex(month_order).fillna(0)
        
            fig.add_trace(go.Scatter(x=avg_data_a.index, y=avg_data_a.values, mode='lines+markers', name=f"{state_selected_a} - Average"))
            fig.add_trace(go.Scatter(x=avg_data_b.index, y=avg_data_b.values, mode='lines+markers', name=f"{state_selected_b} - Average"))
        
            fig.update_layout(title=f'Average Allocations by Month for {state_selected_a} and {state_selected_b} in {start_year}')
    else:
        if plot_type == 'Total Sum':
        # Create yearly line plot for total sum
            summed_data_a = filtered_data_a.groupby('Year')['Allocation'].sum().reset_index()
            summed_data_b = filtered_data_b.groupby('Year')['Allocation'].sum().reset_index()
        
            fig.add_trace(go.Scatter(x=summed_data_a['Year'], y=summed_data_a['Allocation'], mode='lines+markers', name=f"{state_selected_a} - Total Sum"))
            fig.add_trace(go.Scatter(x=summed_data_b['Year'], y=summed_data_b['Allocation'], mode='lines+markers', name=f"{state_selected_b} - Total Sum"))
        
            fig.update_layout(title=f'Comparison of Total Allocations between {state_selected_a} and {state_selected_b} Over the Years')
        else:
        # Create yearly line plot for average
            avg_data_a = filtered_data_a.groupby('Year')['Allocation'].mean().reset_index()
            avg_data_b = filtered_data_b.groupby('Year')['Allocation'].mean().reset_index()
        
            fig.add_trace(go.Scatter(x=avg_data_a['Year'], y=avg_data_a['Allocation'], mode='lines+markers', name=f"{state_selected_a} - Average"))
            fig.add_trace(go.Scatter(x=avg_data_b['Year'], y=avg_data_b['Allocation'], mode='lines+markers', name=f"{state_selected_b} - Average"))
        
            fig.update_layout(title='Average Allocations by LGC Over the Years')

    fig.update_xaxes(title_text='Year' if start_year != end_year else 'Month')
    fig.update_yaxes(title_text='Total Allocation')
    fig.update_layout(legend=dict(x=0, y=1, traceorder='normal'), xaxis=dict(tickangle=90))

    # Display the plot
    st.plotly_chart(fig)



    numeric_columns = lgas.select_dtypes(include=np.number).columns
    lgas[numeric_columns] = lgas[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Function to plot allocations by LGCs
    def plot_allocations_by_lgc(state):
        filtered_data = lgas[lgas['STATE'] == state.capitalize()]

        if filtered_data.empty:
            st.error(f"No data available for state '{state.upper()}'.")
            return None

        # Calculate total allocations by LGC
        total_allocations_by_lgc = filtered_data.set_index('LGC')[numeric_columns].sum(axis=1).reset_index()
        total_allocations_by_lgc.columns = ['LGC', 'Total Allocation']
    
        # Creating the bar plot using Plotly
        fig = px.bar(
        total_allocations_by_lgc.sort_values(ascending=False, by='Total Allocation'),
        x='LGC',
        y='Total Allocation',
        labels={'LGC': 'LGC', 'Total Allocation': 'Total Allocation'},
        title=f'Total Allocations to {state.upper()} (2007-2024)'
    )
        fig.update_layout(xaxis_title='LGC', yaxis_title='Total Allocation', xaxis_tickangle=-90)

        return fig

    # List of unique states
    unique_states = sorted(lgas['STATE'].unique())

    # Streamlit app
    st.title('Allocations to LGCs Per State')

    # Selectbox for states
    state_selected = st.selectbox("Select a state:", unique_states)

    # Plot allocations for the selected state
    fig = plot_allocations_by_lgc(state_selected)
    if fig:
        st.plotly_chart(fig)



    df_melted = df.melt(id_vars=['State', 'Region'], var_name='Date', value_name='Allocation')

# Convert Date column to datetime format
    df_melted['Date'] = pd.to_datetime(df_melted['Date'], format='%Y-%m')

# Extract the year and month from the Date column
    df_melted['Year'] = df_melted['Date'].dt.year
    df_melted['Month'] = df_melted['Date'].dt.strftime('%Y-%m')
    
    

    # Function to extract the month from the date
    def extract_month(date):
        return date.month

    # Function to perform regional analysis with enhanced visualizations
    def regional_analysis(df, region_name):
    # Create a subset of the DataFrame for the specified region
        region_df = df[df['Region'] == region_name]

        # Extract the month from the date
        region_df['Month'] = region_df['Date'].dt.month

    # Aggregating data to get the total allocation per year
        region_agg_year = region_df.groupby(['Year', 'State'])['Allocation'].sum().reset_index()

    # Plot the trend for each state in the region by year
        fig_year = px.line(region_agg_year, x='Year', y='Allocation', color='State', title=f'Total Allocations by State Over the Years in {region_name}')
        fig_year.update_layout(xaxis_title='Year', yaxis_title='Total Allocation', legend_title='State')

        return fig_year

# List of regions in Nigeria
    regions = ['North Central', 'North East', 'North West', 'South East', 'South South', 'South West']

# Streamlit app
    st.title("Region-wise Yearly Analysis of Total Satate Allocations (2007 - 2024)")

# Selectbox for regions
    selected_region = st.selectbox("Select a region:", regions)

# Perform regional analysis and display the plot
    fig_year = regional_analysis(df_melted, selected_region)
    st.plotly_chart(fig_year)
