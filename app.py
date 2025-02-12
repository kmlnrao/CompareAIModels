import streamlit as st
from groq import Groq

client = Groq(api_key=st.secrets.get("GROQ_API_KEY"))

st.title("Compare AI Models", anchor=False)
st.subheader("Compare AI Models Side-by-Side", anchor=False, divider="blue")


# ========== Sidebar ============
st.sidebar.markdown("## Parameters")
st.sidebar.divider()

temp = st.sidebar.slider("Temperature", 0.0, 1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", 0, 8192, value=1024)
stream = st.sidebar.toggle("Stream", value=True)
json_mode = st.sidebar.toggle("JSON Mode", value=False)

st.sidebar.markdown("### Advance Parameters")
st.sidebar.divider()
top_p = st.sidebar.slider("Top P", 0.0, 1.0, value=1.0)
stop_seq = st.sidebar.text_input("Stop Sequence")

# ======== Session Data =========
if "model_a_messages" not in st.session_state:
    st.session_state.model_a_messages = []
    
if "model_b_messages" not in st.session_state:
    st.session_state.model_b_messages = []


# ========= Models UI ================
st.markdown("## Models")

colA, colB = st.columns(2)

with colA:
   st.session_state["model_a"] = st.selectbox(
       "Model A", 
       ["llama3-8b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"], 
       index=0, 
       key=1
    )
   
with colB:
   st.session_state["model_b"] = st.selectbox(
       "Model A", 
       ["llama3-8b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"], 
       index=1, 
       key=2
    )
   
# Session State   
print("SESSION STATE", st.session_state)



# ======= Helpers =============
def handle_user_prompt(newPrompt):
    """Functions adds the user's prompt as an User's message"""
    st.session_state["model_a_messages"].append(
        {"role": "user", "content": newPrompt}
    )
    st.session_state["model_b_messages"].append(
        {"role": "user", "content": newPrompt}
    )
    
def render_messages(model):
    # mode_a_messages
    models_messages = model + "_messages"
    if len(st.session_state[models_messages]) > 0:
        for message in st.session_state[models_messages]:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    

def get_completion(model):
    """Function create a completion from the Groq API using the model, messages and parameters"""
    try:
        models_messages = st.session_state[model + "_messages"]
        
        # API Call
        completion = client.chat.completions.create(
            model=st.session_state[model],
            messages=models_messages,
            stream=stream,
            temperature=temp,
            max_tokens=max_tokens,
            response_format= {"type": "json_object"} if json_mode else {"type": "text"},
            top_p=top_p,
            stop=stop_seq
        )
        
        full_response = ""
        
        if stream: 
            with st.chat_message("assistant"):
                ph = st.empty()
                
                # stream the response to the UI
                for chunk in completion:
                    full_response += chunk.choices[0].delta.content or ""
                    ph.markdown(full_response)
                    
                # add the response to the list
                models_messages.append(
                    {"role": "assistant", "content": full_response}
                )
                
        else:
            with st.spinner("Generating..."):
                with st.chat_message("assistant"):
                    ph = st.empty()
                    ph.write(completion.choices[0].message.content)
                    models_messages.append(
                        {"role": "assistant", 
                        "content": completion.choices[0].message.content
                        }
                    )
        
    except Exception as e:
        print(e)
        st.toast(str(e), icon="🚨")
        
    

# ======== Chat UI ===============
st.divider()
col1, col2 = st.columns(2)

if prompt := st.chat_input("Prompt"):
    st.session_state["prompt"] = prompt
    st.session_state["new_prompt"] = True
    handle_user_prompt(prompt)
else:
    st.session_state["new_prompt"] = False
    


with col1.container(border=True, height=500):
    render_messages("model_a")
    
    if st.session_state["new_prompt"]:
        get_completion("model_a")
    
with col2.container(border=True, height=500):
    render_messages("model_b")
    
    if st.session_state["new_prompt"]:
        get_completion("model_b")
    
    
    
    
print("SESSION STATE", st.session_state)