import streamlit as st
from multiapp import MultiApp
from apps import home, SpatialView, DataView

app = MultiApp()

# st.markdown("""
# # Patterns of spatially refined urban built environment stocks across Chinese cities
# """)

# Add all your application here
st.set_page_config(
        page_title="Patterns of spatially refined urban built environment stocks across Chinese cities",
        page_icon="🌎",
        layout="wide",
        initial_sidebar_state="expanded",
    )

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


app.add_app("Home", home.app)
app.add_app("Spatial Distribution", SpatialView.app)
app.add_app("Building Stock Statistics Data Visualization", DataView.app)

# The main app
app.run()
