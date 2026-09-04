import streamlit as st
HW1 = st.Page('HW/HW1.py', title='HW1', icon=':material/looks_one:')
HW2 = st.Page('HW/HW2.py', title='HW2', icon=':material/looks_two:')
pg = st.navigation([HW2, HW1])  
pg.run()
