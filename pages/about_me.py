import streamlit as st
import numpy as np
from PIL import Image


def app():
    # Title of the main page
    col1, col2 = st.columns(2)
    col1.image("https://raw.githubusercontent.com/Ardy-EDFRE/resource_assessment_tools/main/images/edf_medium_logo.png")
    col2.title("Resource Assessment Tools")

    st.markdown("# 📱 Resource Assessment Tools 📉")

    st.markdown("**ONE STOP SOLUTION FOR ALL YOUR DATA NEEDS**")
    st.markdown("## Introduction")

    st.markdown("## Tool Description")
    st.markdown("A collection of tools that can be used by Wind &amp; Solar teams to analyze specific resource assessments.")

    st.markdown("## Features")

    st.markdown("## 📝 Module-Wise Description")

    st.markdown("The application also uses Streamlit for a multiclass page implementation which can be viewed in the `multipage.py` file. The UI of the application can be seen here. The application is divided into multiple modules, each of which have been described below.")

    st.markdown("![UI of the application](https://raw.githubusercontent.com/Ardy-EDFRE/resource_assessment_tools/main/images/tool_homepage.JPG)")


    st.markdown("_📌 **IEC Terrain Assessment**_ <br/>")


    st.markdown("_📌 **DNV Solar Energy Estimate**_ <br/>")


    st.markdown("## Technology Stack")

    st.markdown("1. Python")
    st.markdown("2. Streamlit")

    st.markdown("## Other Content")

    st.markdown("## 🤝 How to Contribute? [3]")

    st.markdown("- Take a look at the Existing Issues or create your own Issues!")
    st.markdown("- Wait for the Issue to be assigned to you after which you can start working on it.")
    st.markdown("- Fork the Repo and create a Branch for any Issue that you are working upon.")
    st.markdown("- Create a Pull Request which will be promptly reviewed and suggestions would be added to improve it.")
    st.markdown("- Add Screenshots to help us know what this Script is all about.")


    st.markdown("# 👨‍💻 Contributors ✨")

    st.markdown("## Contact")

    st.markdown("any feedback or queries, please reach out to [ardeshir.beheshti@edf-re.com](ardeshir.beheshti@edf-re.com).")

    st.markdown("![Watermark Company Logo](https://raw.githubusercontent.com/Ardy-EDFRE/resource_assessment_tools/main/images/edf_small_logo.png)")