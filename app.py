import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from streamlit_option_menu import option_menu
import numpy as np

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

regions = {
    'North Central': ['Benue', 'Kogi', 'Kwara', 'Nasarawa', 'Niger', 'Plateau', 'Federal Capital Territory'],
    'North East': ['Adamawa', 'Bauchi', 'Borno', 'Gombe', 'Taraba', 'Yobe'],
    'North West': ['Jigawa', 'Kaduna', 'Kano', 'Katsina', 'Kebbi', 'Sokoto', 'Zamfara'],
    'South East': ['Abia', 'Anambra', 'Ebonyi', 'Enugu', 'Imo'],
    'South South': ['Akwa Ibom', 'Bayelsa', 'Cross River', 'Delta', 'Edo', 'Rivers'],
    'South West': ['Ekiti', 'Lagos', 'Ogun', 'Ondo', 'Osun', 'Oyo']
}

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
    st.header("FAAC Allocation")
    st.subheader('')
    st.write("""
    Total tllocations to states since 2007 
    """)
    numeric_columns = df.select_dtypes(include='number')
    total_allocations_by_state = numeric_columns.set_index(df['State']).sum(axis=1).sort_values(ascending=False)
    
    # Set the style
    sns.set_style("whitegrid")
    
    # Create a color palette
    palette = sns.color_palette("viridis", len(total_allocations_by_state))
    
    # Plotting the total allocations by state
    # plt.figure()
    sns.barplot(
        x=total_allocations_by_state.index,
        y=total_allocations_by_state.values,
        palette=palette
    )

    # Customize the plot
    plt.title('Total Allocations by State (2007-2024)', fontsize=18, fontweight='bold')
    plt.xlabel('State', fontsize=14, fontweight='bold')
    plt.ylabel('Total Allocation', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)

    # Remove top and right spines
    sns.despine()

    # Show plot in Streamlit
    st.pyplot(plt)



   # this position is for the cut piece of code
    
    
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

# Streamlit app structure
# with st.sidebar:
#     selected = option_menu(
#         menu_title="Explore",
#         options=["Univariate", "Multivariate"],
#         menu_icon="cast2"
#     )

if selected == "Dynamic":
    st.header("FAAC Allocation")
    st.subheader('')
    st.write("""
    """)

    numeric_columns = df.select_dtypes(include='number')
    total_allocations_by_state = numeric_columns.set_index(df['State']).sum(axis=1).sort_values(ascending=False)

    # Plotting the total allocations by state
  #  plt.figure()
    sns.set_style("whitegrid")
    palette = sns.color_palette("viridis", len(total_allocations_by_state))
    sns.barplot(
        x=total_allocations_by_state.index,
        y=total_allocations_by_state.values,
        palette=palette
    )
    plt.title('Total Allocations by State (2007-2024)', fontsize=18, fontweight='bold')
    plt.xlabel('State', fontsize=14, fontweight='bold')
    plt.ylabel('Total Allocation', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    sns.despine()
    st.pyplot(plt)

    st.subheader('')
    st.write("""
    """)

    selected_states = ['Bayelsa', 'Lagos', 'Akwa Ibom', 'Kano', 'Rivers', 'Delta', 'Ondo']

    # Plotting the monthly allocations trend for selected states
    # plt.figure()
    for state in selected_states:
        dates = df.columns[2:-1]  # Start from index 2 to exclude 'State' and 'Region'
        dates = pd.to_datetime(dates, format='%b-%Y')
        allocations = df.set_index('State').loc[state, dates]
        plt.plot(dates, allocations, label=state)
    plt.title('Allocations Trend (2007-2024) For States with The Most Share of Allocation')
    plt.xlabel('Date')
    plt.ylabel('Monthly Allocation')
    plt.legend()
    plt.xticks(rotation=90)
    st.pyplot(plt)

    st.subheader('')
    st.write("""
    
    """)

    monthly_avg_pivot = pd.pivot_table(
        df_melted,
        values='Allocation',
        index=df_melted['Date'].dt.strftime('%b'),
        columns='Region',
        aggfunc='mean'
    )

    # Define the correct order for months
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Convert index to categorical with specified order
    monthly_avg_pivot.index = pd.Categorical(monthly_avg_pivot.index, categories=month_order, ordered=True)
    monthly_avg_pivot = monthly_avg_pivot.sort_index()

    # Plot monthly average allocations by region
    # plt.figure()
    for region in monthly_avg_pivot.columns:
        plt.plot(monthly_avg_pivot.index, monthly_avg_pivot[region], marker='o', label=region)
    plt.title('Monthly Average Allocations by Region')
    plt.xlabel('Month')
    plt.ylabel('Average Allocation')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    st.subheader('Inflation Trends')
    st.write("""
    
    """)

    # # Pivot table to sum allocations by region for each month
    # pivot_table_sum = pd.pivot_table(
    #     df_melted,
    #     values='Allocation',
    #     index=df_melted['Date'].dt.strftime('%b'),
    #     columns='Region',
    #     aggfunc='sum'
    # )

    # # Pivot table to calculate average allocations by region for each month
    # pivot_table_avg = pd.pivot_table(
    #     df_melted,
    #     values='Allocation',
    #     index=df_melted['Date'].dt.strftime('%b'),
    #     columns='Region',
    #     aggfunc='mean'
    # )


#     date_columns = df.columns[1:-1]  # Adjust this if your date columns are not in this range
#     for col in date_columns:
#         df[col] = pd.to_datetime(df[col], errors='coerce')  # Convert to datetime

#     # Drop rows where date conversion failed (if any)
#     # df.dropna(subset=date_columns, inplace=True)

# # Pivot table to sum allocations by region for each month
#     pivot_table_sum = pd.pivot_table(
#         df,
#         values=date_columns,  # All date columns
#         index='Region',
#         aggfunc='sum'
# )

# # Pivot table to calculate average allocations by region for each month
#     pivot_table_avg = pd.pivot_table(
#         df,
#         values=date_columns,  # All date columns
#         index='Region',
#         aggfunc='mean'
# )



    

    # Transpose the pivot tables
    pivot_table_sum_transposed = pivot_table_sum.transpose()
    pivot_table_avg_transposed = pivot_table_avg.transpose()

    # Set index to datetime for plotting
    pivot_table_sum_transposed.index = pd.to_datetime(pivot_table_sum_transposed.index, format='%b-%Y')
    pivot_table_avg_transposed.index = pd.to_datetime(pivot_table_avg_transposed.index, format='%b-%Y')

    # Plotting sum allocations time series for each region
   # plt.figure()
    for region in pivot_table_sum_transposed.columns:
        plt.plot(pivot_table_sum_transposed[region], label=region)
    plt.title('Sum Allocations Time Series by Region')
    plt.xlabel('Date')
    plt.ylabel('Sum Allocation')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # Plotting average allocations time series for each region
    # plt.figure()
    for region in pivot_table_avg_transposed.columns:
        plt.plot(pivot_table_avg_transposed[region], label=region)
    plt.title('Average Allocations Time Series by Region')
    plt.xlabel('Date')
    plt.ylabel('Average Allocation')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    


   # Melt the DataFrame to long format for easier manipulation
df_melted = df.melt(id_vars=['State', 'Region'], var_name='Date', value_name='Allocation')

# Convert Date column to datetime format
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format='%b-%Y')

st.subheader('Financial and Monetary Indicators')
st.write("""

""")

# Group by region and sum allocations
region_totals = df_melted.groupby('Region')['Allocation'].sum().reset_index()

# Create a pie chart using Plotly
fig1 = px.pie(region_totals, values='Allocation', names='Region',
              title='Total Allocations Share by Region')

# Display the pie chart
st.plotly_chart(fig1)

st.subheader('Credit to Private Sector')
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
    title_font=dict(size=24, family='Arial', color='black'),
    xaxis=dict(tickfont=dict(size=12)),
    yaxis=dict(tickfont=dict(size=12)),
    legend_title_text='Region'
)

# Show the bar plot
st.plotly_chart(fig2)

st.subheader('Download the Entire Report:')

st.header("Visualizations and Insights")
st.markdown('<h4><u>GDP Insights</u></h4>', unsafe_allow_html=True)
st.write("***GDP (constant LCU) Over Time***")

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
    # Filter data by month within the selected year for both states
    filtered_data_a = allocations_by_year[
        (allocations_by_year['State'] == state_selected_a) & 
        (allocations_by_year['Year'] == start_year)
    ]

    if plot_type == "Total Sum":
        # Create monthly line plot for total sum
        summed_data_a = filtered_data_a.groupby('Month')['Allocation'].sum().reindex(month_order).fillna(0)
        fig, ax = plt.subplots()
        ax.plot(summed_data_a.index, summed_data_a.values, marker='o', label=f"{state_selected_a} - Total Sum")
        ax.set_title(f'Total Allocations by Month for {state_selected_a} in {start_year}')
    else:
        # Create monthly line plot for average
        avg_data_a = filtered_data_a.groupby('Month')['Allocation'].mean().reindex(month_order).fillna(0)
        fig, ax = plt.subplots()
        ax.plot(avg_data_a.index, avg_data_a.values, marker='o', label=f"{state_selected_a} - Average")
        ax.set_title(f'Average Allocations by Month for {state_selected_a} in {start_year}')
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
        fig, ax = plt.subplots()
        ax.plot(summed_data_a['Year'], summed_data_a['Allocation'], marker='o', label=f"{state_selected_a} - Total Sum")
        ax.set_title(f'Total Allocations for {state_selected_a} Over the Years')
    else:
        # Create yearly line plot for average
        avg_data_a = filtered_data_a.groupby('Year')['Allocation'].mean().reset_index()
        fig, ax = plt.subplots()
        ax.plot(avg_data_a['Year'], avg_data_a['Allocation'], marker='o', label=f"{state_selected_a} - Average")
        ax.set_title(f'Average Allocations for {state_selected_a} Over the Years')

ax.set_xlabel('Year' if start_year != end_year else 'Month')
ax.set_ylabel('Total Allocation')
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)

st.pyplot(fig)


st.write("""

""")

# Plotting the GDP growth trend
st.write(' ***GDP Trends: Line Plot of GDP Growth (Annual %) Over the Years***')

# Define the order of months for proper plotting
month_order = ['2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06',
               '2023-07', '2023-08', '2023-09', '2023-10', '2023-11', '2023-12']

# Melt the DataFrame to long format for easier manipulation
df_melted = df.melt(id_vars=['State', 'Region'], var_name='Date', value_name='Allocation')

# Convert Date column to datetime format
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format='%Y-%m')

# Extract the year and month from the Date column
df_melted['Year'] = df_melted['Date'].dt.year
df_melted['Month'] = df_melted['Date'].dt.strftime('%Y-%m')

# Now `df_melted` is the DataFrame we'll use
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
        fig, ax = plt.subplots()
        ax.plot(summed_data_a.index, summed_data_a.values, marker='o', label=f"{state_selected_a} - Total Sum")
        ax.plot(summed_data_b.index, summed_data_b.values, marker='o', label=f"{state_selected_b} - Total Sum")
        ax.set_title(f'Total Allocations by Month for {state_selected_a} and {state_selected_b} in {start_year}')
    else:
        # Create monthly line plot for average
        avg_data_a = filtered_data_a.groupby('Month')['Allocation'].mean().reindex(month_order).fillna(0)
        avg_data_b = filtered_data_b.groupby('Month')['Allocation'].mean().reindex(month_order).fillna(0)
        fig, ax = plt.subplots()
        ax.plot(avg_data_a.index, avg_data_a.values, marker='o', label=f"{state_selected_a} - Average")
        ax.plot(avg_data_b.index, avg_data_b.values, marker='o', label=f"{state_selected_b} - Average")
        ax.set_title(f'Average Allocations by Month for {state_selected_a} and {state_selected_b} in {start_year}')
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
        fig, ax = plt.subplots()
        ax.plot(summed_data_a['Year'], summed_data_a['Allocation'], marker='o', label=f"{state_selected_a} - Total Sum")
        ax.plot(summed_data_b['Year'], summed_data_b['Allocation'], marker='o', label=f"{state_selected_b} - Total Sum")
        ax.set_title('Total Allocations by State Over the Years')
    else:
        avg_data_a = filtered_data_a.groupby('Year')['Allocation'].mean().reset_index()
        avg_data_b = filtered_data_b.groupby('Year')['Allocation'].mean().reset_index()
        fig, ax = plt.subplots()
        ax.plot(avg_data_a['Year'], avg_data_a['Allocation'], marker='o', label=f"{state_selected_a} - Average")
        ax.plot(avg_data_b['Year'], avg_data_b['Allocation'], marker='o', label=f"{state_selected_b} - Average")
        ax.set_title('Average Allocations by State Over the Years')

ax.set_xlabel('Year' if start_year != end_year else 'Month')
ax.set_ylabel('Total Allocation')
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)

# Display the plot
st.pyplot(fig)


st.write("""
""")

# Plotting GDP Growth by Regime Type in Nigeria 
st.write('  ***GDP Growth by Regime Type in Nigeria***')
# Sample data for demonstration (replace this with your actual DataFrame)



st.write("""
""")

# Plotting CPI over time
st.write("**Consumer Price Index (CPI) Over Time**")

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
st.title("Allocation Data Analysis")

# Select the first state
state_selected_a = st.selectbox("Select the first LGC:", unique_states, key='state_c')

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
    (allocations_by_year['Year'] <= end_year)
]

fig, ax = plt.subplots()

if start_year == end_year:
    if plot_type == 'Total Sum':
        # Create monthly line plot for total sum
        summed_data_a = filtered_data_a.groupby('Month')['Allocation'].sum().reindex(month_order).fillna(0)
        ax.plot(summed_data_a.index, summed_data_a.values, marker='o', label=f"{state_selected_a} - Total Sum")
        ax.set_title(f'Total Allocations by Month for {state_selected_a} in {start_year}')
    else:
        # Create monthly line plot for average
        avg_data_a = filtered_data_a.groupby('Month')['Allocation'].mean().reindex(month_order).fillna(0)
        ax.plot(avg_data_a.index, avg_data_a.values, marker='o', label=f"{state_selected_a} - Average")
        ax.set_title(f'Average Allocations by Month for {state_selected_a} in {start_year}')
else:
    if plot_type == 'Total Sum':
        # Create yearly line plot for total sum
        summed_data_a = filtered_data_a.groupby('Year')['Allocation'].sum().reset_index()
        ax.plot(summed_data_a['Year'], summed_data_a['Allocation'], marker='o', label=f"{state_selected_a} - Total Sum")
        ax.set_title(f'Total Allocations for {state_selected_a} Over the Years')
    else:
        # Create yearly line plot for average
        avg_data_a = filtered_data_a.groupby('Year')['Allocation'].mean().reset_index()
        ax.plot(avg_data_a['Year'], avg_data_a['Allocation'], marker='o', label=f"{state_selected_a} - Average")
        ax.set_title(f'Average Allocations for {state_selected_a} Over the Years')

ax.set_xlabel('Year' if start_year != end_year else 'Month')
ax.set_ylabel('Total Allocation')
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)

# Display the plot
st.pyplot(fig)

st.write("""

""")

st.markdown('<h4><u>Inflation Insights</u></h4>', unsafe_allow_html=True)

# Plotting Inflation and GDP Growth Rate over time
st.write('***Inflation Rate and GDP Growth Rate Over Time***')

# Filter data based on user selection for second LGC
state_selected_b = st.selectbox("Select the second LGC:", unique_states, key='state_d')

# Filter data based on user selection
filtered_data_b = allocations_by_year[
    (allocations_by_year['LGC'] == state_selected_b) &
    (allocations_by_year['Year'] >= start_year) &
    (allocations_by_year['Year'] <= end_year)
]

fig, ax = plt.subplots()

if start_year == end_year:
    if plot_type == 'Total Sum':
        # Create monthly line plot for total sum
        summed_data_a = filtered_data_a.groupby('Month')['Allocation'].sum().reindex(month_order).fillna(0)
        summed_data_b = filtered_data_b.groupby('Month')['Allocation'].sum().reindex(month_order).fillna(0)
        ax.plot(summed_data_a.index, summed_data_a.values, marker='o', label=f"{state_selected_a} - Total Sum")
        ax.plot(summed_data_b.index, summed_data_b.values, marker='o', label=f"{state_selected_b} - Total Sum")
        ax.set_title(f'Total Allocations by Month for {state_selected_a} and {state_selected_b} in {start_year}')
    else:
        # Create monthly line plot for average
        avg_data_a = filtered_data_a.groupby('Month')['Allocation'].mean().reindex(month_order).fillna(0)
        avg_data_b = filtered_data_b.groupby('Month')['Allocation'].mean().reindex(month_order).fillna(0)
        ax.plot(avg_data_a.index, avg_data_a.values, marker='o', label=f"{state_selected_a} - Average")
        ax.plot(avg_data_b.index, avg_data_b.values, marker='o', label=f"{state_selected_b} - Average")
        ax.set_title(f'Average Allocations by Month for {state_selected_a} and {state_selected_b} in {start_year}')
else:
    if plot_type == 'Total Sum':
        # Create yearly line plot for total sum
        summed_data_a = filtered_data_a.groupby('Year')['Allocation'].sum().reset_index()
        summed_data_b = filtered_data_b.groupby('Year')['Allocation'].sum().reset_index()
        ax.plot(summed_data_a['Year'], summed_data_a['Allocation'], marker='o', label=f"{state_selected_a} - Total Sum")
        ax.plot(summed_data_b['Year'], summed_data_b['Allocation'], marker='o', label=f"{state_selected_b} - Total Sum")
        ax.set_title('Total Allocations by LGC Over the Years')
    else:
        # Create yearly line plot for average
        avg_data_a = filtered_data_a.groupby('Year')['Allocation'].mean().reset_index()
        avg_data_b = filtered_data_b.groupby('Year')['Allocation'].mean().reset_index()
        ax.plot(avg_data_a['Year'], avg_data_a['Allocation'], marker='o', label=f"{state_selected_a} - Average")
        ax.plot(avg_data_b['Year'], avg_data_b['Allocation'], marker='o', label=f"{state_selected_b} - Average")
        ax.set_title('Average Allocations by LGC Over the Years')

ax.set_xlabel('Year' if start_year != end_year else 'Month')
ax.set_ylabel('Total Allocation')
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)

# Display the plot
st.pyplot(fig)
