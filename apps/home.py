import streamlit as st


def app():
    st.markdown('## Background')
    st.markdown('Rapid urbanization and booming construction activities exert resource and environmental pressures for global sustainability, especially for cities in developing countries. By estimating urban built environment stocks for 50 selected Chinese cities with big data mining and analytics techniques, we found their stock quantities are positively correlated with urban socioeconomic.')
    st.markdown('We developed a website for visualizing the spatial distribution of built-environment stocks across Chinese cities and their relationship to socioeconomic status to better understand the results discussed in the article **Patterns of spatially refined urban built environment stocks across Chinese cities**.')
    st.markdown("This project is based on [streamlit](https://streamlit.io/), all the source code and data can be found in [Github](https://github.com/CUGbaoyi/Streamlit-Built-Environment-Stocks)")
    st.markdown("## Introduction")
    st.markdown("This project include two main function")
    st.markdown("""
        1. Spatial Distribution
        2. Building Stock Statistics Data Visualization
        """)
    st.markdown("### Spatial Distribution")
    st.markdown("You can choose the city and material you want to visualize on this page (Default values in Beijing and built-environment stocks). Click Start to start drawing.")
    st.markdown("**Attention! For this step, it may take few seconds to tens of seconds to complete. The greater the amount of data in the city, the slower the processing speed will be.**")
    st.image("https://i.loli.net/2021/06/22/VD5co8h7H4pujeI.png", caption="Data visualize example", width=700)

    st.markdown("### Building Stock Statistics Data Visualization")
    st.markdown("As we discovered in the article, there are some relationships between the stock of buildings and economic and social characteristics. We visualized this relationship in each city.")
    st.markdown("There are 5 images in each city, which are:")
    st.markdown("""
        1. The proportion of various materials in the building MS (interactive image)
        2. The ccdf curves for building MS 
        3. The Lorentz curves for building MS
        4. The correlations between the arranged population and building MS at grid level (interactive image)
        5. The pdf curves for BtR ratio (interactive image)
        """)
    st.image("https://i.loli.net/2021/06/22/ev43QBa6CLHRn1f.png", caption="The proportion of various materials and ccdf curves example", width=700)
    st.image("https://i.loli.net/2021/06/22/Q7bleykEvWc2SHN.png", caption="The gini index and the relationship between population and building MS example", width=700)
    st.image("https://i.loli.net/2021/06/22/TVx3DWJHMfadYp4.png", caption="The BtR ration example", width=700)

    st.markdown("**All conclusions in the article are reproducible, and the specific code can be found in [Github](https://github.com/CUGbaoyi/Streamlit-Built-Environment-Stocks). If you have any questions, you can contact me or raise an issue in [Github](https://github.com/CUGbaoyi/Streamlit-Built-Environment-Stocks)**")

