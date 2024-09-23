#my imported packages
#huggingface-cli upload Esben922/Assignment . . --repo-type=space 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
import streamlit as st
import altair as alt

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
    st.warning("Please select a Country from the sidebar ⚠️")
    st.stop()
filtered_df = df[df['country'] == selected_country]
filtered_df = filtered_df.merge(df2[['id', 'Loan Theme ID', 'Loan Theme Type', 'Partner ID']], how='left', left_on='id', right_on='id')
df1_unique_partners = df1[['Partner ID', 'Field Partner Name']].drop_duplicates(subset=['Partner ID'])
filtered_df = filtered_df.merge(df1_unique_partners, how='left', left_on='Partner ID', right_on='Partner ID')



#Siderbar filter: Region
loan_theme_types = filtered_df['Loan Theme Type'].unique().tolist()
selected_themes = st.sidebar.multiselect("Select Loan Theme Types 🏢", loan_theme_types, default=loan_theme_types)
if not selected_themes:
    st.warning("Please select a region from the sidebar ⚠️")
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