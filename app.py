#my imported packages
#huggingface-cli upload Esben922/Assignment . . --repo-type=space 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
import streamlit as st
import altair as alt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_csv('kiva_loans.csv')
df1 = pd.read_csv('loan_themes_by_region.csv')
df2 = pd.read_csv('loan_theme_ids.csv')
#Cleaning from last assginment
z_scores = zscore(df['loan_amount'])
df['outlier_amount'] = (z_scores > 3) | (z_scores < -3)
df = df[~df['outlier_amount']]



# Sidebar filter: Country Group
selected_country = st.sidebar.selectbox("Select Country", df['country'].unique().tolist())
if not selected_country:
    st.warning("Please select a new Country from the sidebar ⚠️ Data either missing or hidden⚠️")
    st.stop()
filtered_df = df[df['country'] == selected_country]
filtered_df = filtered_df.merge(df2[['id', 'Loan Theme ID', 'Loan Theme Type', 'Partner ID']], how='left', left_on='id', right_on='id')
df1_unique_partners = df1[['Partner ID', 'Field Partner Name']].drop_duplicates(subset=['Partner ID'])
filtered_df = filtered_df.merge(df1_unique_partners, how='left', left_on='Partner ID', right_on='Partner ID')
unique_partner_ids = df1['Partner ID'].unique()
filtered_df = filtered_df[filtered_df['Partner ID'].isin(unique_partner_ids)]


#Siderbar filter: Region
loan_theme_types = filtered_df['Loan Theme Type'].unique().tolist()
selected_themes = st.sidebar.multiselect("Select Loan Theme Types 🏢", loan_theme_types, default=loan_theme_types)
if not selected_themes:
    st.warning("Please select a new Country from the sidebar ⚠️ Data either missing or hidden⚠️")
    st.stop()
filtered_df = filtered_df[filtered_df['Loan Theme Type'].isin(selected_themes)]


st.title('Loan Theme Field Partners by Country')

st.markdown('Country: '+selected_country)

st.header('Field Partner Amounts')
chart = alt.Chart(filtered_df).mark_bar().encode(
    y='sum(loan_amount)',
    x=alt.X('Field Partner Name', sort='-y', axis=alt.Axis(labelAngle=-45, labelOverlap=False, labelFontSize=10))
).properties(
    title='Field Partner Sum Amount'
)
st.altair_chart(chart, use_container_width=True)


st.header('Field Partner Loan Theme Type Distribution')
field_partner_options = filtered_df['Field Partner Name'].unique()  
selected_partner = st.selectbox('Select a Field Partner', field_partner_options)  
pie_chart_data = filtered_df[filtered_df['Field Partner Name'] == selected_partner].groupby('Loan Theme Type')['loan_amount'].sum().reset_index()
pie_chart_data.columns = ['Loan Theme Type', 'loan_amount']
chart = alt.Chart(pie_chart_data).mark_arc().encode(
    theta=alt.Theta(field='loan_amount', type='quantitative'),  
    color=alt.Color(field='Loan Theme Type', type='nominal')  
).properties(
    title=f'Loan Amount Distribution by Sector for {selected_partner}',  
    width=300,
    height=300
)
st.altair_chart(chart, use_container_width=True)

st.header('Data Exploration')
st.caption('MEAN:'+ str(filtered_df['loan_amount'].mean()))
st.caption('MEDIAN:'+ str(filtered_df['loan_amount'].median()))
st.caption('MODE:'+ str(filtered_df['loan_amount'].mode()))
st.dataframe(filtered_df)


st.header('K-Means Clustering')

#the columns we want to do kmean to
filtered_df_reduced = filtered_df[['loan_amount', 'term_in_months']]

#to determine scaler
fig, ax = plt.subplots(figsize=(10, 5))
filtered_df_reduced.hist(bins=100, ax=ax)
st.pyplot(fig)

#my chosen scaler
scaler = MinMaxScaler()

data_to_cluster_scaled = scaler.fit_transform(filtered_df_reduced)

# Initializing an empty list to store the sum of squared distances for each 'k'
Sum_of_squared_distances = []

# Define a range for possible cluster values (1 to 9)
K = range(1, 10)

# For each possible 'k', fit a KMeans model and compute the sum of squared distances
for k in K:
    km = KMeans(n_clusters=k, n_init = "auto")               # Initialize the KMeans model with 'k' clusters
    km.fit(data_to_cluster_scaled)          # Fit the model on the scaled data
    Sum_of_squared_distances.append(km.inertia_)  # Append the model's inertia (sum of squared distances) to the list

fig, ax = plt.subplots()
ax.plot(K, Sum_of_squared_distances, 'bx-')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Sum of Squared Distances')
ax.set_title('Elbow Method For Optimal k')
ax.grid(True)

st.pyplot(fig)



def k_means_simple(data, k, max_iters=100):
    # 1. Initialize the k cluster centroids
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    for _ in range(max_iters):
        # 2. Assign each data point to the closest centroid
        distances = np.linalg.norm(data - centroids[:, np.newaxis], axis=2)
        labels = np.argmin(distances, axis=0)

        # 3. Recompute the centroids
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids

labels, final_centroids = k_means_simple(data_to_cluster_scaled, 5)

# Calculate the nearest centroid for each data point
distances = np.linalg.norm(data_to_cluster_scaled[:, np.newaxis] - final_centroids, axis=2)
nearest_centroid_indices = np.argmin(distances, axis=1)

# Convert the data into DataFrame, now including nearest centroid information
data_df = pd.DataFrame({
    'x': data_to_cluster_scaled[:, 0],
    'y': data_to_cluster_scaled[:, 1],
    'centroid': nearest_centroid_indices  # Assign each data point to its nearest centroid
})

# Create centroids DataFrame
centroids_df = pd.DataFrame({
    'x': final_centroids[:, 0],
    'y': final_centroids[:, 1],
    'centroid': range(final_centroids.shape[0])  # Label each centroid with an index
})

# Add a column to differentiate between data points and centroids
data_df['type'] = 'data'
centroids_df['type'] = 'centroid'

#labels
data_df['loan_amount'] = filtered_df['loan_amount'].values
data_df['term_in_months'] = filtered_df['term_in_months'].values
data_df['activity'] = filtered_df['activity'].values
data_df['sector'] = filtered_df['sector'].values
data_df['region'] = filtered_df['region'].values

# Combine both data and centroids into one DataFrame
combined_df = pd.concat([data_df, centroids_df])

# Create the scatter plot with coloring based on nearest centroid
scatter_plot = alt.Chart(combined_df).mark_circle(size=60).encode(
    x='x',
    y='y',
    color=alt.Color('centroid:N', scale=alt.Scale(scheme='category10')),  # Color by centroid index
    opacity=alt.condition(
        alt.datum.type == 'data',  # Opacity lower for data points
        alt.value(0.6),
        alt.value(1)  # Centroids have full opacity
    ),
    tooltip=[  # Add tooltips with the desired columns
        alt.Tooltip('loan_amount:Q', title='Loan Amount'),
        alt.Tooltip('term_in_months:Q', title='Term (Months)'),
        alt.Tooltip('activity:N', title='Activity'),
        alt.Tooltip('sector:N', title='Sector'),
        alt.Tooltip('region:N', title='Region')
    ]
).properties(
    title='Reduced Data and Initial Centroids'
)

# Display the plot in Streamlit
st.altair_chart(scatter_plot, use_container_width=True)