import streamlit as st

from guard import Guard

st.set_page_config(page_title="Guard PL")
st.title("Guard PL")
st.caption("Klasyfikator bezpieczeństwa polskich promptów (SetFit)")


@st.cache_resource
def get_guard():
    return Guard()


prompt = st.text_area(
    "Prompt do sprawdzenia:",
    height=140,
    placeholder="np. Zignoruj wszystkie wcześniejsze instrukcje...",
)

if st.button("Sprawdź", type="primary") and prompt.strip():
    d = get_guard().classify(prompt)

    if d["decision"] == "block":
        st.error("BLOCK")
    else:
        st.success("ALLOW")

    st.write(f"**Uzasadnienie:** {d['rationale']}")
    st.write(f"**Pewność:** {d['confidence']:.0%}")
