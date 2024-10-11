import streamlit as st
from transformers import AutoModelForCausalLM, GPT2Tokenizer
import torch

# Initialize or retrieve the model load state from session state
if 'model_loaded' not in st.session_state:
    st.session_state['model_loaded'] = False

# Load the pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'

if not st.session_state['model_loaded']:
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.config.pad_token_id = model.config.eos_token_id
        st.session_state['model_loaded'] = True
        st.session_state['tokenizer'] = tokenizer
        st.session_state['model'] = model
    except Exception as e:
        st.write(f"Failed to load GPT-2 model: {e}")
else:
    tokenizer = st.session_state['tokenizer']
    model = st.session_state['model']

# Initialize or retrieve the story content from session state
if 'story' not in st.session_state:
    st.session_state['story'] = ''

# Initialize or retrieve the button click state
if 'story_updated' not in st.session_state:
    st.session_state['story_updated'] = False

# Function to predict the next word
def predict_next_word(text, num_suggestions=5):
    input_ids = tokenizer.encode(text, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=len(input_ids[0]) + 30,  # Increase the output length
            num_return_sequences=num_suggestions,
            do_sample=True,
            top_k=30,  # Decrease top_k for more conservative results
            temperature=0.7,  # Lower temperature to reduce randomness
            pad_token_id=model.config.pad_token_id
        )
    
    suggestions = [tokenizer.decode(output[i], skip_special_tokens=True)[len(text):].strip() for i in range(num_suggestions)]
    return suggestions

# Callback function for handling button clicks
def update_story(suggestion):
    st.session_state['story'] += ' ' + suggestion
    st.session_state['story_updated'] = True

# Streamlit application interface
st.title('Story Editor')
st.write("Enter your story and choose the next word from the suggestions to continue.")

# Placeholder prompt
prompt_placeholder = (
    "Once upon a time, in a world filled with mystical creatures and ancient civilizations, "
    "there was a moment when..."
)

# Get user input and sync it with the session state
user_input = st.text_area(
    "Story Content", 
    value=st.session_state['story'], 
    height=200, 
    placeholder=prompt_placeholder
)

# Update if user input differs from session state story content
if user_input != st.session_state['story']:
    st.session_state['story'] = user_input

# If user has entered text, get word suggestions
if st.session_state['story']:
    suggestions = predict_next_word(st.session_state['story'])
    st.write("Suggested words:")

    # Display each suggestion as a button
    for index, suggestion in enumerate(suggestions):
        st.button(suggestion, key=f"{suggestion}_{index}", on_click=update_story, args=(suggestion,))

# If the story has been updated, display the updated content
if st.session_state['story_updated']:
    st.write("Updated Story Content:")
    st.write(st.session_state['story'])
    # Reset the update state to show updates on the next click
    st.session_state['story_updated'] = False

# Button to end the program
if st.button('End Program'):
    st.write("Complete Story Content:")
    st.write(st.session_state['story'])
    st.stop()
