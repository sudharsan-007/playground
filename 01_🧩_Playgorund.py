import streamlit as st


if __name__ == "__main__":
    st.set_page_config(page_title="Sudu Playgound", page_icon="🕹️", layout="wide") # Create a page icon as img and add
    st.title("# WELCOME TO MY PROJECT PAGE 🎈")
    
    st.markdown("""
                Welcome to my project website. I love Robotics, Machine learning and cars, and of course you can see that trend 
                continuing with all my projects. This website is a barebone conceptual project page with lots of bugs and 
                issues(at least I admit 😏). I am constantly trying to add features without breaking the code like all companies do. 
                I am currenlty working to add reinforcement learning games to add to this page. And also some data science 
                projects on the side. \n
                **What are you waiting for click on a project in the side bar and play with it**
                """) 
    
    st.caption("Note: Running calculations takes time when using cheapest server, so be patient for things to load. ")
    
    st.header("About me")
    
    st.markdown("""
                **Name:**         Sudharsan Ananth <br> 
                **Nick Name:**    SUDO (get it? Linux+Me)  <br> 
                **Age:**          Still young, yea very 23 now <br> 
                **Profession:**   Why do u think i am making these, cuz I am jobless and I need one <br> 
                **Currently learning:** MOJO (Its a very new language, thats probably going to change the world) <br> 
                **Description:**  A passionated python developer 🐍 , I dont write pythonic code, i am better C'ick code 😅<br> 
                **Professional info:** Please check my [LinkedIn](https://www.linkedin.com/in/sudharsan-ananth/)  <br> 
                **Code editor of choice**: VScode and VIM (Learning) ✏️
                """, unsafe_allow_html=True)
    
    st.header("My Vision 👀")
    
    st.markdown("""
                As I described above, this page is a testing bed for something bigger. I want to create a website 
                with a lot of games like flappy bird Hill climb racing built into web interface and then give all the machine 
                learning tools such that anyone can train Machine learning models with a web interface with no code. I also want
                implement Model Predictive Controller, PID controller and iLQR for simpler problems. I want this website to be 
                interactive giving ability to tune any parameters(Including changing no of hidden layers, increasing neurons etc). 
                In the future I also want to get code inputs from user to run custom functions, costs or even models. 
                If you want are excited please feel to ping me in my [LinkedIn](https://www.linkedin.com/in/sudharsan-ananth/) 
                and we can collabrate. 
                """)
    
    st.subheader("How to find me 📍") 
    
    st.markdown("""
                **Where do I live:** New York babe 🗽 (yes its famous for lots 🐀 and 🚇) \n 
                **Email:** sudharsan.ananth@gmail.com 📧 \n 
                **LinkedIn:** [https://www.linkedin.com/in/sudharsan-ananth/](https://www.linkedin.com/in/sudharsan-ananth/) 🤓 \n 
                """)
