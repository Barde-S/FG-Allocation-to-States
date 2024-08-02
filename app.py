import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
import numpy as np
from nbconvert import HTMLExporter
import nbformat
import os

# Load data
df = pd.read_excel('FAAC DATA - Data Community.xlsx', sheet_name='State')
df = df.iloc[:, :-7]

lgas = pd.read_excel('FAAC DATA - Data Community.xlsx', sheet_name='LGA')


for col in lgas.columns:
    if lgas[col].isna().sum() == 776:
        lgas = lgas.drop(columns=col) # Remove inplace=True and assignment


for col in lgas[2:]:
    if lgas[col].isna().sum() > 0:
        # Attempt to convert the column to numeric, coercing errors to NaN
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
        options=["Univariate", "Multivariate"],
        menu_icon="cast"
    )


if selected == "Univariate":
    st.header("Exploratory Data Analysis")
    st.subheader('GDP Insights')
    st.write("""
    GDP in Constant and Current LCU
    - Correlation: The GDP in constant and current Local Currency Units (LCU) shows a very high correlation (0.994), indicating consistency in GDP measurements over time.
    - Trend: A significant upward trend in GDP highlights sustained economic growth in Nigeria over the years.
    """)
    numeric_columns = df.select_dtypes(include='number')
    total_allocations_by_state = numeric_columns.set_index(df['State']).sum(axis=1).sort_values(ascending=False)
    
    # Set the style
    sns.set_style("whitegrid")
    
    # Create a color palette
    palette = sns.color_palette("viridis", len(total_allocations_by_state))
    
    # Plotting the total allocations by state
    plt.figure(figsize=(20, 8))
    sns.barplot(
    x=total_allocations_by_state.index,
    y=total_allocations_by_state.values,
    palette=palette)

    # Customize the plot
    plt.title('Total Allocations by State (2007-2024)', fontsize=18, fontweight='bold')
    plt.xlabel('State', fontsize=14, fontweight='bold')
    plt.ylabel('Total Allocation', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)

    # Remove top and right spines
    sns.despine()

    # Show plot
    plt.show()


   # this position is for the cut piece of code
    
    
    regions = {
    'North Central': ['Benue', 'Kogi', 'Kwara', 'Nasarawa', 'Niger', 'Plateau', 'Federal Capital Territory'],
    'North East': ['Adamawa', 'Bauchi', 'Borno', 'Gombe', 'Taraba', 'Yobe'],
    'North West': ['Jigawa', 'Kaduna', 'Kano', 'Katsina', 'Kebbi', 'Sokoto', 'Zamfara'],
    'South East': ['Abia', 'Anambra', 'Ebonyi', 'Enugu', 'Imo'],
    'South South': ['Akwa Ibom', 'Bayelsa', 'Cross River', 'Delta', 'Edo', 'Rivers'],
    'South West': ['Ekiti', 'Lagos', 'Ogun', 'Ondo', 'Osun', 'Oyo']}
    
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
    aggfunc='mean')




      

    st.subheader('Population Dynamics')
    st.write("""
    Total, Female, and Male Populations
    - Correlation: Strong correlations exist between the total, female, and male populations, indicating consistent growth across all demographic segments.
    - Trend: Steady population growth from 1960 onwards, with annual growth rates around 2% for both male and female populations.
    """)
    # Select a few states for trend visualization
    selected_states = ['Bayelsa', 'Lagos', 'Akwa Ibom', 'Kano', 'Rivers', 'Delta', 'Ondo']

    # Plotting the monthly allocations trend for selected states
    plt.figure(figsize=(22, 8))
    for state in selected_states:
        # Extract the dates (column names excluding 'State' and 'Region') and corresponding allocations for the current state
        dates = df.columns[2:-1]  # Start from index 2 to exclude 'State' and 'Region'
        dates = pd.to_datetime(dates, format='%b-%Y') 
        allocations = df.set_index('State').loc[state, dates]
        plt.plot(dates, allocations, label=state) 
    plt.title('Allocations Trend (2007-2024) For States with The Most Share of Allocation')
    plt.xlabel('Date')
    plt.ylabel('Monthly Allocation')
    plt.legend()
    plt.xticks(rotation=90)
    plt.show()

    st.subheader('Petrol Prices and GDP')
    st.write("""
    - Correlation: Petrol prices in Naira show a positive correlation with GDP in US dollars (0.229) and per capita GDP (0.286), suggesting a relationship between energy prices and economic performance.
    - Trend: Petrol prices have seen a steady increase, particularly from the late 1990s onwards, correlating with global oil price changes and domestic policy adjustments.
    """)
    monthly_avg_pivot = pd.pivot_table(
    df_melted,
    values='Allocation',
    index=df_melted['Date'].dt.strftime('%b'),
    columns='Region',
    aggfunc='mean')

    # Define the correct order for months
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Convert index to categorical with specified order
    monthly_avg_pivot.index = pd.Categorical(monthly_avg_pivot.index, categories=month_order, ordered=True)
    monthly_avg_pivot = monthly_avg_pivot.sort_index()

    # Plot monthly average allocations by region
    plt.figure(figsize=(22, 8))
    for region in monthly_avg_pivot.columns:
        plt.plot(monthly_avg_pivot.index, monthly_avg_pivot[region], marker='o', label=region)
    plt.title('Monthly Average Allocations by Region')
    plt.xlabel('Month')
    plt.ylabel('Average Allocation')
    plt.legend()
    plt.grid(True)
    plt.show()

    st.subheader('Inflation Trends')
    st.write("""
    - 1981: Extremely high inflation rates following the global oil price collapse.
    - 1988-1989: High consumer price inflation, reflecting economic instability.
    - 1992-1995: Persistent high inflation due to structural adjustment programs and economic reforms.
    """)
   # Pivot table to sum allocations by region for each month
    pivot_table_sum = pd.pivot_table(
    df,
    values=df.columns[1:-1],  # All date columns
    index='Region',
    aggfunc='sum')



# Pivot table to calculate average allocations by region for each month
    pivot_table_avg = pd.pivot_table(df,
        values=df.columns[1:-1],  # All date columns
        index='Region',
        aggfunc='mean')



# Transpose the pivot tables
    pivot_table_sum_transposed = pivot_table_sum.transpose()
    pivot_table_avg_transposed = pivot_table_avg.transpose()

# Set index to datetime for plotting
    pivot_table_sum_transposed.index = pd.to_datetime(pivot_table_sum_transposed.index, format='%b-%Y')
    pivot_table_avg_transposed.index = pd.to_datetime(pivot_table_avg_transposed.index, format='%b-%Y')


# Plotting sum allocations time series for each region
    plt.figure(figsize=(22, 8))
    for region in pivot_table_sum_transposed.columns:
        plt.plot(pivot_table_sum_transposed[region], label=region)
    plt.title('Sum Allocations Time Series by Region')
    plt.xlabel('Date')
    plt.ylabel('Sum Allocation')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plotting average allocations time series for each region
    plt.figure(figsize=(22, 8))
    for region in pivot_table_avg_transposed.columns:
        plt.plot(pivot_table_avg_transposed[region], label=region)
    plt.title('Average Allocations Time Series by Region')
    plt.xlabel('Date')
    plt.ylabel('Average Allocation')
    plt.legend()
    plt.grid(True)
    plt.show()



    st.subheader('Financial and Monetary Indicators')
    st.write("""
    - Total Reserves: A significant increase in total reserves over the years, indicating improved accumulation of foreign exchange and gold reserves.
    - Narrow Money and Money Supply (M3): Both metrics show noticeable upward trends, reflecting increased monetary circulation.
    """)
    # Group by region and sum allocations
    region_totals = melted_df.groupby('Region')['Allocation'].sum().reset_index()

    fig = px.pie(region_totals, values='Allocation', names='Region',
                 title='Total Allocations Share by Region')

    # Display the chart
    fig.show()

    st.subheader('Credit to Private Sector')
    st.write("""
    - Credit Growth: Steady increase in credit extended to the private sector, indicating growth in private sector activities and investments.
    """)
    top_states_by_region = df_melted.loc[df_melted.groupby('Region')['Allocation'].idxmax()]

    # Sort the result by Allocation
    top_states_by_region = top_states_by_region.sort_values(by='Allocation', ascending=False)

      # Create the bar plot using Plotly
    fig = px.bar(
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
    fig.update_layout(
        autosize=False,
        width=1650,
        height=600,
        xaxis_title='State',
        yaxis_title='Total Allocation',
        xaxis_tickangle=-90,
        title_font=dict(size=24, family='Arial', color='black'),
        xaxis=dict(tickfont=dict(size=12)),
        yaxis=dict(tickfont=dict(size=12)),
        legend_title_text='Region'
)

    # Show the plot
    fig.show()

    st.subheader('Financial and Monetary Indicators')
    st.write("""
    - Total Reserves: A significant increase in total reserves over the years, indicating improved accumulation of foreign exchange and gold reserves.
    - Narrow Money and Money Supply (M3): Both metrics show noticeable upward trends, reflecting increased monetary circulation.
    """)
    # Group by region and sum allocations
    region_totals = melted_df.groupby('Region')['Allocation'].sum().reset_index()



    st.subheader('Download the Entire Report:')

    with open("./Media/eda_report.pdf", "rb") as pdf_file:
        PDFbyte = pdf_file.read()
    
    st.download_button(
        label="Download Full Report",
        data=PDFbyte,
        file_name="EDA_Full_Report.pdf",
        mime="application/pdf"
    )

elif selected == "Multivariate":
    st.header("Visualizations and Insights")

    
    st.markdown('<h4><u>GDP Insights</u></h4>', unsafe_allow_html=True)
    # Plot GDP (constant LCU)
    st.write(" ***GDP (constant LCU) Over Time***")
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
    state_selected_a = st.selectbox("Select the first State:", unique_states)

    # Prompt user to select the year range
    start_year, end_year = st.slider("Select a year range:", min(unique_years), max(unique_years), (min(unique_years), max(unique_years)))

    # Prompt user to select whether they want the total sum or average
    plot_type = st.radio("Select the type of plot:", ("Total Sum", "Average"))

    # Check if start year and end year are the same
    if start_year == end_year:
        # Filter data by month within the selected year for both states
        filtered_data_a = allocations_by_year[
            (allocations_by_year['State'] == state_selected_a) & 
            (allocations_by_year['Year'] == start_year)
    ]

        if plot_type == "Total Sum":
        # Create monthly line plot for total sum
            summed_data_a = filtered_data_a.groupby('Month')['Allocation'].sum().reindex(month_order).fillna(0)
            fig, ax = plt.subplots(figsize=(22, 8))
            ax.plot(summed_data_a.index, summed_data_a.values, marker='o', label=f"{state_selected_a} - Total Sum")
            ax.set_title(f'Total Allocations by Month for {state_selected_a} in {start_year}')
        else:
            # Create monthly line plot for average
            avg_data_a = filtered_data_a.groupby('Month')['Allocation'].mean().reindex(month_order).fillna(0)
            fig, ax = plt.subplots(figsize=(22, 8))
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
            fig, ax = plt.subplots(figsize=(22, 8))
            ax.plot(summed_data_a['Year'], summed_data_a['Allocation'], marker='o', label=f"{state_selected_a} - Total Sum")
            ax.set_title(f'Total Allocations for {state_selected_a} Over the Years')
        else:
        # Create yearly line plot for average
            avg_data_a = filtered_data_a.groupby('Year')['Allocation'].mean().reset_index()
            fig, ax = plt.subplots(figsize=(22, 8))
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
From 1960 to 2020, the GDP (constant LCU) shows a strong upward trend, reflecting substantial long-term economic growth. Notable growth began in the 2000s, highlighting significant economic expansion and a positive trajectory.
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
    state_selected_a = st.selectbox("Select the first State:", unique_states)

    # Prompt user to select the second state
    state_selected_b = st.selectbox("Select the second State:", unique_states)

    # Prompt user to select the year range
    start_year, end_year = st.slider("Select a year range:", min(unique_years), max(unique_years), (min(unique_years), max(unique_years)))

    # Prompt user to select whether they want the total sum or average
    plot_type = st.radio("Select the type of plot:", ("Total Sum", "Average"))

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
            fig, ax = plt.subplots(figsize=(22, 8))
            ax.plot(summed_data_a.index, summed_data_a.values, marker='o', label=f"{state_selected_a} - Total Sum")
            ax.plot(summed_data_b.index, summed_data_b.values, marker='o', label=f"{state_selected_b} - Total Sum")
            ax.set_title(f'Total Allocations by Month for {state_selected_a} and {state_selected_b} in {start_year}')
        else:
            # Create monthly line plot for average
            avg_data_a = filtered_data_a.groupby('Month')['Allocation'].mean().reindex(month_order).fillna(0)
            avg_data_b = filtered_data_b.groupby('Month')['Allocation'].mean().reindex(month_order).fillna(0)
            fig, ax = plt.subplots(figsize=(22, 8))
            ax.plot(avg_data_a.index, avg_data_a.values, marker='o', label=f"{state_selected_a} - Average")
            ax.plot(avg_data_b.index, avg_data_b.values, marker='o', label=f"{state_selected_b} - Average")
            ax.set_title(f'Average Allocations by Month for {state_selected_a} and {state_selected_b} in {start_year}')
    else:
    # Filter data based on user selection for yearly plot
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
            fig, ax = plt.subplots(figsize=(22, 8))
            ax.plot(summed_data_a['Year'], summed_data_a['Allocation'], marker='o', label=f"{state_selected_a} - Total Sum")
            ax.plot(summed_data_b['Year'], summed_data_b['Allocation'], marker='o', label=f"{state_selected_b} - Total Sum")
            ax.set_title('Total Allocations by State Over the Years')
         else:
        # Create yearly line plot for average
            avg_data_a = filtered_data_a.groupby('Year')['Allocation'].mean().reset_index()
            avg_data_b = filtered_data_b.groupby('Year')['Allocation'].mean().reset_index()
            fig, ax = plt.subplots(figsize=(22, 8))
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
The annual GDP growth rate has shown significant volatility, with sharp increases and decreases. A notable peak occurred in the early 1970s, indicating rapid economic expansion. The mid-1980s and early 1990s experienced negative growth, reflecting economic recessions. Since the 2000s, the growth rate has been more stable despite some fluctuations. Overall, the data highlights the economy's resilience and ability to recover and continue growing.
""")

#plotting GDP Growth by Regime Type in Nigeria 
    st.write('  ***GDP Growth by Regime Type in Nigeria***')
    fig, ax = plt.subplots(figsize=(10, 6))
    data.groupby('Regime Type')['GDP growth (annual %)'].mean().plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title('GDP Growth by Regime Type in Nigeria ')
    ax.set_xlabel('Regime Type')
    ax.set_ylabel('Average GDP Growth Rate (%)')
    st.pyplot(fig)
    st.write("""
This bar plot shows the average GDP growth rate by regime type in Nigeria from 1960 to 1978. 
It compares the GDP growth under different regimes (e.g., Military and Civilian) during this period.
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
    unique_states = [lgc.upper() for lgc in allocations_by_year['LGC'].unique()]
    unique_years = allocations_by_year['Year'].unique()

# Streamlit App
    st.title("Allocation Data Analysis")

# Select the first state
    state_selected_a = st.selectbox("Select the first LGC:", unique_states)

# Select the year range
    start_year, end_year = st.select_slider(
    "Select the year range:",
        options=unique_years,
        value=(min(unique_years), max(unique_years))
)

# Select the type of plot
    plot_type = st.radio(
    "Select the type of plot:",
    ('Total Sum', 'Average')
)

# Filter data based on user selection
    filtered_data_a = allocations_by_year[
        (allocations_by_year['LGC'] == state_selected_a) &
        (allocations_by_year['Year'] >= start_year) &
        (allocations_by_year['Year'] <= end_year)
]

    fig, ax = plt.subplots(figsize=(22, 8))

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
Nigeria's Consumer Price Index (CPI) shows a sharp increase, especially from the early 2000s onwards. 
There was a relatively stable and low CPI level until the late 1980s, after which it began to rise. 
The sharp increase in CPI indicates significant inflationary pressures in recent decades, particularly post-2000.
""")
    st.markdown('<h4><u>Inflation Insights</u></h4>', unsafe_allow_html=True)
# Plotting Inflation and GDP Growth Rate over time
    st.write('***Inflation Rate and GDP Growth Rate Over Time***')
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
    unique_states = [lgc.upper() for lgc in allocations_by_year['LGC'].unique()]
    unique_years = allocations_by_year['Year'].unique()

# Streamlit App
    st.title("Allocation Data Analysis")

# Select the first state
    state_selected_a = st.selectbox("Select the first LGC:", unique_states)

# Select the second state
    state_selected_b = st.selectbox("Select the second LGC:", unique_states)

    # Select the year range
    start_year, end_year = st.select_slider(
    "Select the year range:",
        options=unique_years,
        value=(min(unique_years), max(unique_years))
)

# Select the type of plot
    plot_type = st.radio(
    "Select the type of plot:",
    ('Total Sum', 'Average')
)

# Filter data based on user selection
    filtered_data_a = allocations_by_year[
        (allocations_by_year['LGC'] == state_selected_a) &
        (allocations_by_year['Year'] >= start_year) &
        (allocations_by_year['Year'] <= end_year)
]
    filtered_data_b = allocations_by_year[
        (allocations_by_year['LGC'] == state_selected_b) &
        (allocations_by_year['Year'] >= start_year) &
        (allocations_by_year['Year'] <= end_year)
]

    fig, ax = plt.subplots(figsize=(22, 8))

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


#     st.write("""
# There is significant volatility in inflation and GDP growth rates from 1960 to 1980. 
# This period likely experienced economic instability or transitions. Notable spikes in inflation rate are observed in the late 1970s and early 1980s. 
# These periods correspond to historical global economic events like the oil crises, which led to high inflation. 
# After the 1990s, inflation and GDP growth rates appeared less volatile, indicating more stable financial conditions.
# """)
    
# # Plotting the Consumer Price Index and Inflation trend
#     st.write(' ***Consumer Price Index and Inflation over the Years***')
#     fig, ax = plt.subplots(figsize=(10, 6))  
#     sns.lineplot(x='Year', y='Consumer price index (2010 = 100)', data=data, marker='o', label='Consumer price index (2010 = 100)', linewidth=2.5, ax=ax)
#     sns.lineplot(x='Year', y='Inflation, consumer prices (annual %)', data=data, marker='o', label='Inflation, consumer prices (annual %)', linewidth=2.5, ax=ax,color='red')
#     ax.set_title('Consumer Price Index and Inflation over the Years')
#     ax.set_xlabel('Year')
#     ax.set_ylabel('Index / Inflation (%)')
#     ax.legend()
#     st.pyplot(fig)
#     st.write("""
# - Consumer Price Index (CPI, blue line): The CPI shows a steady increase over time, with a sharp rise starting around 2000, indicating a significant increase in the general price level of goods and services.
# - Inflation Rate (red line): The inflation rate fluctuates more compared to the CPI, with notable spikes around the 1970s and early 1980s, indicating periods of high inflation. Afterward, the inflation rate remains relatively stable, generally below 10% annually.
# """)
    
    
# # Plotting Comparison of GDP Growth, Inflation, and Crude Oil Price
#     st.write('***Comparison of GDP Growth, Inflation, and Crude Oil Price***')
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.lineplot(x='Year', y='GDP growth (annual %)', data=data, marker='o', label='GDP Growth (annual %)', color='green', ax=ax)
#     sns.lineplot(x='Year', y='Inflation, consumer prices (annual %)', data=data, marker='o', label='Inflation (annual %)',  color='red',ax=ax)
#     sns.lineplot(x='Year', y='Crude oil price(per barrel in $)', data=data, marker='o', label='Crude Oil Price ($)', color='orange', ax=ax)
#     ax.set_xlabel('Year')
#     ax.set_ylabel('Rate / Price')
#     ax.set_title('Comparison of GDP Growth, Inflation, and Crude Oil Price over Time')
#     ax.legend()
#     st.pyplot(fig)
#     st.write("""
# This plot compares the annual GDP growth rate, inflation rate, and crude oil price from 1960 to 2021.
# - **GDP Growth (annual %)**: Shows the percentage change in GDP year-over-year & stable with minor fluctuations, mostly between 0% and 10%.
# - **Inflation (annual %)**: Represents the annual increase in consumer prices .It Shows significant spikes around the 1970s and 1980s, indicating periods of high inflation.
# - **Crude Oil Price ($)**: Displays the price per barrel of crude oil in US dollars.Relatively stable until around 1970, then shows a sharp increase starting in the early 2000s, peaking around 2008, and remaining volatile thereafter
# """)
    
#     st.markdown('<h4><u>Oil Sector and Petrol Prices</u></h4>', unsafe_allow_html=True)
# # Plotting Trade Openness Index and Crude Oil Price Over Time 
#     st.write('***Trade Openness Index and Crude Oil Price over the Years***')
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.lineplot(x='Year', y='Trade Openness Index(%)', data=data, marker='o', label='Trade Openness Index (%)', ax=ax)
#     sns.lineplot(x='Year', y='Crude oil price(per barrel in $)', data=data, marker='o', label='Crude Oil Price (per barrel in $)', ax=ax)
#     ax.set_title('Trade Openness Index and Crude Oil Price over the Years')
#     ax.set_xlabel('Year')
#     ax.set_ylabel('Index / Price ($)')
#     ax.legend()
#     plt.xticks(rotation=45)
#     st.pyplot(fig)
#     st.write("""
# - **The Trade Openness Index (%)** is relatively stable from the 1960s to the mid-1970s, after which it shows an increasing trend with fluctuations.
# - **The Crude Oil Price (per barrel in $)** : remained low and stable until the early 1970s, then experienced a sharp increase during the oil crisis of the late 1970s.
# - **Both indices** exhibit noticeable spikes and drops, reflecting economic and geopolitical events.
# """)
# # Plotting Petrol Price (Naira)
#     st.write('***Petrol Prices(Nira) over Years***')
#     fig_petrol, ax_petrol = plt.subplots(figsize=(10, 6))
#     ax_petrol.plot(data['Year'], data['Petrol Price (Naira)'], marker='o')
#     ax_petrol.set_xlabel('Year')
#     ax_petrol.set_ylabel('Petrol Price (Naira)')
#     ax_petrol.set_title('Petrol Price (Naira) Over Time')
#     st.pyplot(fig_petrol)
#     st.write("""
# The Line plot illustrates the 'Petrol Price (Naira)' trend over time. It reflects changes in petrol prices in Nigeria.
# """) 

#     st.markdown('<h4><u>Population Dynamics</u></h4>', unsafe_allow_html=True)
# # Plotting Total Population Over Time 
#     st.write('***Total Population Over Time*** ')
#     fig, ax = plt.subplots(figsize=(12, 8))
#     ax.bar(data['Year'], data['Population, total'], color='skyblue')
#     ax.set_xlabel('Year')
#     ax.set_ylabel('Population')
#     ax.set_title('Total Population Over Time (Bar Chart)')
#     st.pyplot(fig)
#     st.write("""
# This bar chart shows the total population over time from 1960 to 2020. Each bar represents the population count for a specific year, providing a visual comparison of population growth.
# """)
# #plotting Population Growth by Regime Over Time
#     st.write('***Population Growth by Regime Over Time***')
#     fig, ax = plt.subplots(figsize=(10, 6))
#     for regime_type, regime_data in data.groupby('Regime Type'):
#          ax.plot(regime_data['Year'], regime_data['Population, total'], marker='o', label=regime_type)
#     ax.set_xlabel('Year')
#     ax.set_ylabel('Population')
#     ax.set_title('Population Growth by Regime Over Time')
#     ax.legend()
#     st.pyplot(fig)
#     st.markdown("""
# This plot shows the population growth by regime type over time. Each line represents the population growth under different regimes (e.g., Military,Monarchy,Presidency) from 1960 to 2021.
# """)
    
#     st.markdown('<h4><u>Monteray Policy & Lending Interest</u></h4>', unsafe_allow_html=True)
# # Plotting -Line plots of Monetary Indicators      
#     st.write('***Monetary Indicators Over Time***')
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.lineplot(x='Year', y='Money Supply M3', data=data, marker='o', label='Money Supply M3', ax=ax)
#     sns.lineplot(x='Year', y='Base Money', data=data, marker='o', label='Base Money', ax=ax)
#     sns.lineplot(x='Year', y='Currency in Circulation', data=data, marker='o', label='Currency in Circulation', ax=ax)
#     ax.set_xlabel('Year')
#     ax.set_ylabel('Amount')
#     ax.set_title('Monetary Indicators Over Time')
#     ax.legend()
#     st.pyplot(fig)
#     st.write("""
# This plot shows the trends of key monetary indicators over time:
# - **Money Supply M3**: Total amount of money in circulation, including cash and bank deposits.
# - **Base Money**: Total amount of money issued by the central bank.
# - **Currency in Circulation**: Total amount of physical currency in circulation.

# The plot illustrates how these indicators have changed over the years, providing insights into the monetary policy and economic conditions.
# """)
  
# # Ploting  Lending Interest Rate over Time
#     st.write('***Lending Interest Rate Over Time***')
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.plot(df['Year'], df['Lending interest rate (%)'], marker='o')
#     ax.set_title('Lending Interest Rate Over Time')
#     ax.set_xlabel('Year')
#     ax.set_ylabel('Interest Rate (%)')
#     ax.grid(True)
#     plt.xticks(rotation=45)
#     st.pyplot(fig)
#     st.markdown("""
# - The lending interest rate over time shows the fluctuation and trends in the cost of borrowing.
# - Significant peaks and troughs may correspond to economic policies, financial crises, and market conditions.
# """) 
    
#     st.markdown('<h4><u>Sector Performance</u></h4>', unsafe_allow_html=True)  
# # Plotting Agricultural Production Over Time 
#     st.write('**Agricultural Production Over Time***')
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.plot(df['Year'], df['Crop production'], marker='o', label='Crop production')
#     ax.plot(df['Year'], df['Livestock'], marker='o', label='Livestock')
#     ax.plot(df['Year'], df['Forestry'], marker='o', label='Forestry')
#     ax.plot(df['Year'], df['Fishing'], marker='o', label='Fishing')
#     ax.set_title('Agricultural Production Over Time')
#     ax.set_xlabel('Year')
#     ax.set_ylabel('Production Volume')
#     ax.legend()
#     ax.grid(True)
#     plt.xticks(rotation=45)
#     st.pyplot(fig)
#     st.markdown("""
# This plot shows the agricultural production trends over time, including crop production, livestock, forestry, and fishing.
# """)
# # Plotting Sector Values Over Time
#     st.write('**Sector Values Over Time**')
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.plot(df['Year'], df['Construction Sector Naira(million)'], marker='o', label='Construction')
#     ax.plot(df['Year'], df['Information and Communication Sector (millions in naira)'], marker='o', label='Info & Comm')
#     ax.plot(df['Year'], df['Real Estate (millions of naira)'], marker='o', label='Real Estate')
#     ax.set_title('Sector Values Over Time')
#     ax.set_xlabel('Year')
#     ax.set_ylabel('Value in Naira (millions)')
#     ax.legend()
#     ax.grid(True)
#     plt.xticks(rotation=45)
#     st.pyplot(fig)
#     st.markdown("""
# - **Construction Sector**: The value of the construction sector has shown substantial growth, particularly post-2000.
# - **Information and Communication Sector**: This sector has also grown significantly, especially from the early 2000s.
# - **Real Estate**: The real estate sector has shown a steady increase in value, mirroring trends in urbanization and real estate development.
# """)   

#     st.markdown('<h4><u>Unemployment & Debt Analysis</u></h4>', unsafe_allow_html=True)
#     # Unemployment Rates Comparison
#     st.write('***Unemployment Rates Comparison***')
#     fig_unemployment, ax_unemployment = plt.subplots(figsize=(10, 6))
#     ax_unemployment.plot(df1['Year'], df1['Unemployment, male (% of male labor force)'], marker='o', label='Male Unemployment Rate (%)')
#     ax_unemployment.plot(df1['Year'], df1['Unemployment, female (% of female labor force)'], marker='o', label='Female Unemployment Rate (%)')
#     ax_unemployment.set_title('Unemployment Rates Comparison Over Time')
#     ax_unemployment.set_xlabel('Year')
#     ax_unemployment.set_ylabel('Unemployment Rate (%)')
#     ax_unemployment.legend()
#     ax_unemployment.grid(True)
#     st.pyplot(fig_unemployment)
#     st.write("""
# This chart shows unemployment trends for males and females from 2011 to 2021. Initially similar, the gap between male and female unemployment rates widened after 2011. Male unemployment peaked at 5.4% in 2019, while female unemployment reached 6.4% in 2021. By 2023, female unemployment decreased, but male rates stayed high. This indicates a correlation in unemployment trends across genders.
# """)


#     #plotting Export Rate (NGN/USD) and Debt % GDP Over Time 
#     st.write('**Export Rate (NGN/USD) and Debt % GDP Over Time**')
#     fig_export_debt, ax_export_debt = plt.subplots(figsize=(10, 6))
#     ax_export_debt.plot(df1['Year'], df1['Export Rate (NGN/USD)'], marker='o', label='Export Rate (NGN/USD)', color='blue')
#     ax_export_debt.plot(df1['Year'], df1['Debt % GDP'], marker='o', label='Debt % GDP', color='red')
#     ax_export_debt.set_title('Export Rate (NGN/USD) and Debt % GDP Over Time')
#     ax_export_debt.set_xlabel('Year')
#     ax_export_debt.set_ylabel('Rate')
#     ax_export_debt.legend()
#     ax_export_debt.grid(True)
#     st.pyplot(fig_export_debt)
#     st.write("""
# - The Chart Shows trends in Export Rate (NGN/USD) and Debt % GDP over the years.
# -  As export rates increased, debt percentage also rose.
# - Increased exports have had minimal impact on reducing national debt, suggesting inefficiencies in using export revenues for debt management.
# """)


# # Plotting Debt % GDP Over Time
#     st.write('***Debt % GDP Over Time***')
#     fig_debt, ax_debt = plt.subplots(figsize=(10, 6))
#     ax_debt.plot(df1['Year'], df1['Debt % GDP'], marker='o', label='Debt % GDP')
#     ax_debt.set_title('Debt % GDP Over Time')
#     ax_debt.set_xlabel('Year')
#     ax_debt.set_ylabel('Debt % GDP')
#     ax_debt.legend()
#     ax_debt.grid(True)
#     st.pyplot(fig_debt)
#     st.write("This chart shows the trend of Debt % GDP over the years. From 1990 to 2023, Nigeria's debt percentage has fluctuated. It was high in the early 1990s, declined until 1999, then peaked in the late 1990s. From 2006 to 2010, debt percentage significantly reduced, likely due to better economic management or debt relief. However, since 2011, the debt percentage has been rising, raising concerns about the sustainability of Nigeria's fiscal policies and economic health.")

#     st.markdown('<h4><u>Government Expenditure</u></h4>', unsafe_allow_html=True)
# #Plotting *Nigerian Government Expenditure Analysis
#     st.write("**Nigerian Government Expenditure Analysis**")
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.plot(df2["Year"], df2["Nigerian Government Expenditure"], marker='o', linestyle='-', color='b')
#     ax.set_title('Nigerian Government Expenditure for each Year')
#     ax.set_xlabel('Year', fontsize=12)
#     ax.set_ylabel('Nigerian Government Expenditure', fontsize=12)
#     st.pyplot(fig)
#     st.write("""
# - Government expenditure generally shows a declining trend from 2000 to around 2006, followed by fluctuations with no clear upward or downward trend until 2015.
# - Expenditure remained relatively stable from 2016 to 2021, with slight fluctuations.
# - In 2022 and 2023, there was a slight increase in expenditure, possibly indicating economic adjustments or policy changes.
# - A notable increase in 2024 suggests a potential shift in fiscal policy or economic priorities.
# """)
# # Plotting Nigerian Government Expenditure and Government Debt Analysis
#     st.write("***Nigerian Government Expenditure and Government Debt Analysis***")
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.plot(df2["Year"], df2["Nigerian Government Expenditure"], marker='o', linestyle='-', color='b', label='Expenditure')
#     ax.plot(df2["Year"], df2["Government Debt (% of GDP)"], marker='o', linestyle='-', color='r', label='Debt (% of GDP)')
#     ax.set_title('Nigerian Government Expenditure and Government Debt (% of GDP)')
#     ax.set_xlabel('Year', fontsize=12)
#     ax.set_ylabel('Values', fontsize=12)
#     ax.legend()
#     st.pyplot(fig)
#     st.write("""
# - **Government Expenditure**: 
#   - Shows a declining trend from 2000 to around 2006.
#   - Fluctuates without a clear trend from 2006 to 2015.
#   - Remains relatively stable from 2016 to 2021 with slight fluctuations.
#   - Slight increase in 2022 and 2023, possibly indicating economic adjustments or policy changes.
#   - Notable increase in 2024 suggests a potential shift in fiscal policy or economic priorities.
# - **Government Debt (% of GDP)**:
#   - Displays a continuous declining trend from 2000 to 2024.
#   - Indicates efforts to manage and reduce national debt over the years.
# """) 


# # Continue with content for Consumer Price Index
# elif selected == "Consumer Price Index":
  
#     # Continue with content for Consumer Price Index

#     # Load the data
#     data = pd.read_csv(file_path)

#     # Define the features and target
#     independent_feature = 'Consumer price index (2010 = 100)'
#     dependent_features = [
#         'GDP (current LCU)', 'Official exchange rate (LCU per US$, period average)', 'Population, total',
#         'Cumulative crude oil production up to and including year', 'Narrow Money', 'Credit to Private Sector',
#         'Demand Deposits', 'Population ages 65 and above (% of total population)', 'Money Supply M2',
#         'Population, female', 'Quasi Money', 'Bank Reserves', 'Livestock production index (2014-2016 = 100)',
#         'Net Foreign Assets', 'GDP (constant LCU)'
#     ]

#     X = data[dependent_features]
#     y = data[independent_feature]

#     # Split the data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Initialize the CatBoostRegressor
#     model = CatBoostRegressor(verbose=0)

#     # Train the model
#     model.fit(X_train, y_train)

#     # Save the model to disk
#     pickle.dump(model, open('./model.sav', 'wb'))
#     print("Model saved as 'model.sav'")

#     # Load the model from disk
#     model = pickle.load(open('./model.sav', 'rb'))

#     # Streamlit app
#     st.title('Consumer Price Index Prediction')

#     # Input fields for user to enter feature values
#     inputs = {}
#     for feature in dependent_features:
#         inputs[feature] = st.number_input(f'Enter {feature}', value=0.0)

#     # Predict button
#     if st.button('Predict'):
#         # Create a DataFrame for the inputs
#         input_data = pd.DataFrame([inputs])

#         # Make prediction
#         prediction = model.predict(input_data)

#         # Display the prediction
#         st.write(f'The predicted Consumer Price Index (2010 = 100) is: {prediction[0]}')

  
# elif selected == "GDP":
   
#     # Continue with content for GDP

#     # Load your data
#     data = pd.read_csv(file_path)

#     # Independent variables
#     independent_vars = [
#         "Year",
#         "Consumer price index (2010 = 100)",
#         "GDP (current LCU)",
#         "Inflation, GDP deflator (annual %)",
#         "Official exchange rate (LCU per US$, period average)",
#         "Total reserves (includes gold, current US$)",
#         "Population, total",
#         "Population ages 15-64 (% of total population)",
#         "Money Supply M3",
#         "Base Money",
#         "Currency in Circulation",
#         "Bank Reserves",
#         "Currency Outside Banks",
#         "Quasi Money",
#         "Other Assets Net",
#         "CBN Bills",
#         "Special Intervention Reserves",
#         "GDPBillions of US $",
#         "Per CapitaUS $",
#         "Petrol Price (Naira)"
#     ]

#     # Target variable
#     target_var = "GDP per capita (current US$)"

#     # Define features and target
#     X = data[independent_vars]
#     y = data[target_var]

#     # Handle missing values (if any)
#     X.fillna(X.mean(), inplace=True)
#     y.fillna(y.mean(), inplace=True)

#     # Scale features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # Split the data
#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#     # Create and train the model
#     model = Lasso(alpha=0.1)  # Adjust alpha as needed
#     model.fit(X_train, y_train)

#     # Evaluate the model
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)

#     # Save the model and scaler
#     with open('lasso_model.pkl', 'wb') as f:
#         pickle.dump(model, f)

#     with open('scaler.pkl', 'wb') as f:
#         pickle.dump(scaler, f)

#     # Streamlit app
#     st.title('GDP per Capita Prediction')

#     # Input fields for independent variables
#     year = st.number_input('Year')
#     cpi = st.number_input('Consumer price index (2010 = 100)')
#     gdp_lcu = st.number_input('GDP (current LCU)')
#     inflation = st.number_input('Inflation, GDP deflator (annual %)')
#     exchange_rate = st.number_input('Official exchange rate (LCU per US$, period average)')
#     reserves = st.number_input('Total reserves (includes gold, current US$)')
#     population_total = st.number_input('Population, total')
#     population_age_15_64 = st.number_input('Population ages 15-64 (% of total population)')
#     money_supply_m3 = st.number_input('Money Supply M3')
#     base_money = st.number_input('Base Money')
#     currency_in_circulation = st.number_input('Currency in Circulation')
#     bank_reserves = st.number_input('Bank Reserves')
#     currency_outside_banks = st.number_input('Currency Outside Banks')
#     quasi_money = st.number_input('Quasi Money')
#     other_assets_net = st.number_input('Other Assets Net')
#     cbn_bills = st.number_input('CBN Bills')
#     special_intervention_reserves = st.number_input('Special Intervention Reserves')
#     gdp_billions = st.number_input('GDPBillions of US $')
#     per_capita_usd = st.number_input('Per CapitaUS $')
#     petrol_price = st.number_input('Petrol Price (Naira)')

#     # Create input data
#     input_data = np.array([[year, cpi, gdp_lcu, inflation, exchange_rate, reserves,
#                             population_total, population_age_15_64, money_supply_m3,
#                             base_money, currency_in_circulation, bank_reserves,
#                             currency_outside_banks, quasi_money, other_assets_net,
#                             cbn_bills, special_intervention_reserves, gdp_billions,
#                             per_capita_usd, petrol_price]])

#     # Scale input data
#     input_data_scaled = scaler.transform(input_data)

#     # Predict
#     if st.button('Predict GDP per Capita'):
#         prediction = model.predict(input_data_scaled)
#         st.write(f'Predicted GDP per Capita: ${prediction[0]:,.2f}')

# elif selected == "Notebook Viewer":
#     st.title("Jupyter Notebooks Viewer & Downloader")
#     notebooks_dir = 'Notebooks'
#     if not os.path.exists(notebooks_dir):
#         st.error(f"The directory {notebooks_dir} does not exist.")
#     else:
#         notebook_files = [f for f in os.listdir(notebooks_dir) if f.endswith('.ipynb')]
#         selected_notebook = st.selectbox("Select a notebook to display", notebook_files)

#         if selected_notebook:
#             notebook_path = os.path.join(notebooks_dir, selected_notebook)
#             try:
#                 notebook_html = convert_notebook_to_html(notebook_path)
#                 st.components.v1.html(notebook_html, height=1000, scrolling=True)
#                 st.download_button(
#                     label="Download Notebook",
#                     data=open(notebook_path, "rb").read(),
#                     file_name=selected_notebook,
#                     mime="application/octet-stream"
#                 )
#             except Exception as e:
#                 st.error(f"An error occurred while displaying the notebook: {e}")

# elif selected == "Contact Us":
#     st.header("Contact Us")
#     st.image("./Media/contactus.jpg", width = 500)
#     st.subheader("We'd love to hear from you!")
#     st.write("""
#     Here are the ways you can reach out to our team:
    
#     - **Email:** [omdenakdnachapter@gmail.com](mailto:omdenakdnachapter@gmail.com)
#     - **Phone:** +234 701 041 2114
#     - **Website:** [Omdena Kaduna Chapter](https://www.omdena.com/local-chapters/kaduna-nigeria-chapter)
#     """)
