# Custom imports
from multipage import MultiPage
# import your pages here
from pages import iec_v2, table_edits_aggrid, dnv_solar, about_me, solar_comparisons

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
app.add_page("Solar Estimate Comparisons", solar_comparisons.app)
app.add_page("Geoportal Data", geoportal_data.app)
# The main app
app.run()
