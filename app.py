import os
import streamlit as st
import numpy as np
from PIL import  Image

# Custom imports 
from multipage import MultiPage
from pages import iec_v2, dnv_solar, about_me # import your pages here

# Create an instance of the app 
app = MultiPage()

# Title of the main page
display = Image.open('edf_logo.jpg')
display = np.array(display)
# st.image(display, width = 400)
# st.title("Data Storyteller Application")
col1, col2 = st.columns(2)
col1.image(display, width = 400)
col2.title("Resource Assessment Tools")

# Add all your application here
app.add_page("About the page", about_me.app)
app.add_page("IEC Terrain Assessment", iec_v2.app)
app.add_page("DNV Solar Energy Estimate", dnv_solar.app)
# The main app
app.run()
