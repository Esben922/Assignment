#my imported packages
#huggingface-cli upload Esben922/Assignment . . --repo-type=space 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
import streamlit as st
import altair as alt

df = pd.read_csv('loan_themes_by_region.csv')
#Cleaning from last assginment
df.drop(['Partner ID','Loan Theme ID','forkiva','geocode_old','ISO', 'geocode','names','geo','mpi_region','mpi_geo','rural_pct'], axis="columns", inplace=True)
z_scores = zscore(df['amount'])
df['outlier_amount'] = (z_scores > 3) | (z_scores < -3)
df = df[~df['outlier_amount']]



# Sidebar filter: Country Group
selected_country = st.sidebar.selectbox("Select Country", df['country'].unique().tolist())
if not selected_country:
    st.warning("Please select a Country from the sidebar ⚠️")
    st.stop()
filtered_df = df[df['country'] == (selected_country)]

#Siderbar filter: Region
loan_theme_types = filtered_df['Loan Theme Type'].unique().tolist()
selected_themes = st.sidebar.multiselect("Select Loan Theme Types 🏢", loan_theme_types, default=loan_theme_types)
if not selected_themes:
    st.warning("Please select a region from the sidebar ⚠️")
    st.stop()
filtered_df = filtered_df[filtered_df['Loan Theme Type'].isin(selected_themes)]


st.title('Assignment Dashboard')

st.markdown('Country: '+selected_country)

st.caption('MEAN:'+ str(filtered_df['amount'].mean()))
st.caption('MEAN:'+ str(filtered_df['amount'].median()))
st.caption('MODE:'+ str(filtered_df['amount'].mode()))
st.dataframe(filtered_df)

chart = alt.Chart(filtered_df).mark_bar().encode(
    y='sum(amount)',
    x=alt.X('Field Partner Name', sort='-y', axis=alt.Axis(labelAngle=-45, labelOverlap=False, labelFontSize=10))
).properties(
    title='Field Partner Sum Amount'
)
st.altair_chart(chart, use_container_width=True)

field_partner_options = filtered_df['Field Partner Name'].unique()  
selected_partner = st.selectbox('Select a Field Partner', field_partner_options)  

pie_chart_data = filtered_df[filtered_df['Field Partner Name'] == selected_partner].groupby('Loan Theme Type')['amount'].sum().reset_index()
pie_chart_data.columns = ['Loan Theme Type', 'amount']

chart = alt.Chart(pie_chart_data).mark_arc().encode(
    theta=alt.Theta(field='amount', type='quantitative'),  
    color=alt.Color(field='Loan Theme Type', type='nominal')  
).properties(
    title=f'Loan Amount Distribution by Sector for {selected_partner}',  
    width=300,
    height=300
)

st.altair_chart(chart, use_container_width=True)