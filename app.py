import streamlit as st

search_input = st.text_input("Search", key='search')

col1, col2 = st.columns(2)
with col1:
    search_button = st.button('Search')
with col2:
    clear_button = st.button('Clear')