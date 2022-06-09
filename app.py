# Custom imports
from multipage import MultiPage
# import your pages here
from pages import iec_v2, dnv_solar, about_me

import streamlit as st
import matplotlib.pyplot as plt

plt.style.use("dark_background")
st.set_page_config(layout="wide")

# Create an instance of the app 
app = MultiPage()

# Add all your application here
app.add_page("About the page", about_me.app)
app.add_page("IEC Terrain Assessment", iec_v2.app)
app.add_page("DNV Solar Energy Estimate", dnv_solar.app)
# The main app
app.run()
