import streamlit as st
import requests
import uuid

st.set_page_config(page_title="What Beats Rock?", page_icon="🪨")

st.title("🪨 What Beats Rock?")

if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())
if 'game_over' not in st.session_state:
    st.session_state['game_over'] = False

st.markdown("Start with the word **rock**. Enter what you think beats it!")

persona = st.radio("Choose a host persona:", ["cheery", "serious"])
guess = st.text_input("Your guess:")

if st.button("Submit") and guess and not st.session_state['game_over']:
    res = requests.post("http://localhost:8000/guess", json={
        "session_id": st.session_state['session_id'],
        "guess": guess,
        "persona": persona
    })
    data = res.json()

    st.markdown(f"**Score:** {data['score']}")
    if data['result'] == "correct":
        st.success(data['message'])
        st.balloons()
    elif data['result'] == "incorrect":
        st.warning(data['message'])
    elif data['result'] == "duplicate":
        st.error(data['message'])
        st.session_state['game_over'] = True
        st.snow()

    if data.get("last_guesses"):
        st.markdown("**Last 5 Guesses:**")
        st.write(data["last_guesses"])

    if data.get("global_count"):
        st.markdown(f"**Guessed globally**: {data['global_count']} times")
