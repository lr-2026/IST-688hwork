import streamlit as st
HW1 = st.Page('HW/HW1.py', title='HW1', icon=':material/looks_one:')
HW2 = st.Page('HW/HW2.py', title='HW2', icon=':material/looks_two:')
HW3 = st.Page('HW/HW3.py', title='HW3', icon=':material/looks_3:')
pg = st.navigation([HW3, HW2, HW1])  
pg.run()
