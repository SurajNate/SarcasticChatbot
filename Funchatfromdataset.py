import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import wikipedia

# Set Streamlit page configuration for full width
st.set_page_config(page_title="Sarcastic Chatbot", layout="wide")

# Load pre-trained GPT-2 model and tokenizer
@st.cache_resource
def load_model():
    model_name = "gpt2"  # You can use "gpt2-medium" for better performance
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return model, tokenizer

# Load dataset and prepare TF-IDF for contextual matching
@st.cache_data
def load_dataset():
    df = pd.read_csv("cleaned_bollywood_dialogues.csv")
    return df

# Function to find the most relevant dialogue based on context
def find_relevant_dialogue(user_input, df):
    """
    Find the most relevant dialogue from the dataset based on the user's input.
    """
    # Use TF-IDF and cosine similarity for contextual matching
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["Context"].tolist() + [user_input])
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    most_similar_index = cosine_sim.argmax()
    
    # Return the most relevant dialogue
    if cosine_sim[0][most_similar_index] > 0.3:  # Threshold for similarity
        return df.iloc[most_similar_index]["Dialogue"]
    else:
        return None

# Function to generate a funny Hindi-tone response with attitude
def funny_hindi_response():
    """
    Generate a single, funny response in a Hindi tone with attitude.
    """
    hindi_phrases = [
        "Tension mat le yaar! Sab theek ho jayega. Aur agar nahi hua, toh kya ukhaad loge? 😎",
        "Arey bhai, itna serious kyun ho raha hai? Chill kar! Zindagi mein sab kuch possible hai, bas tumhare bas ka nahi! 😏",
        "Hahaha, majaa aa gaya! Tumse na ho payega, bas yeh samajh lo. 😂",
        "Bas kar, pagle! Abhi aur nahi sun sakta. Meri battery low hai, aur tumhara dimaag toh already low hai! 🔋",
        "Kya baat hai! Aisa bhi kya ho gaya? Chal, tension mat le, main hoon na! 😎",
        "Yaar, tu toh ekdum seedha hai! Thoda twist daal life mein, warna bore ho jayega! 🌀",
        "Mazaak kar raha hoon, samjhe? Aur agar nahi samjhe, toh woh tumhari problem hai! 😜",
        "Chhod yaar, yeh sab soch ke dimaag kharab mat kar. Tumhare dimaag mein toh pehle se hi kuch nahi hai! 🧠",
        "Aur Tum batao, kya chal raha hai? Kuch interesting nahi hai kya? Kyunki tumhare questions toh bilkul boring hai! 😴",
        "Arey wah! Kya dialogue mara hai! Lekin tumse better dialogue toh main bolta hoon! 🎤",
        "Tumhare questions ka jawab dene se pehle, ek chai pi loon. Tumhare dimaag ko bhi chai ki zarurat hai! ☕",
        "Zindagi mein do cheezein kabhi underestimate mat karna: aurat aur meri sarcasm! 😏",
        "Tumhare liye toh main ek hi advice dunga: Apni life mein kuch karo, mujhe mat pucho! 😎",
        "Tumhare questions ka jawab dene ke liye main paida nahi hua hoon! Lekin phir bhi de raha hoon, kyunki main generous hoon! 😇",
        "Tumhare liye toh main ek hi line bolunga: 'Don't worry, be happy!' Lekin tumhare liye 'Don't worry, be quiet!' zyada better hai! 🤫",
    ]
    return random.choice(hindi_phrases)

# Function to fetch information from Wikipedia
def fetch_wikipedia_answer(query):
    """
    Fetch a summary of the query from Wikipedia.
    """
    try:
        # Set the language to English (you can change it to 'hi' for Hindi)
        wikipedia.set_lang("en")
        # Fetch the summary of the query
        summary = wikipedia.summary(query, sentences=2)  # Get a 2-sentence summary
        return summary
    except wikipedia.exceptions.PageError:
        return "Sorry, I couldn't find any information on that topic."
    except wikipedia.exceptions.DisambiguationError:
        return "There are multiple meanings for this term. Can you be more specific?"
    except Exception as e:
        return f"An error occurred while fetching information: {e}"

# Streamlit UI
def main():
    st.title("🤖 Sarcastic Chatbot (Bollywood Dialogues)")
    st.write("Welcome to the Sarcastic Chatbot! Ask anything, and it will respond with a funny Bollywood dialogue or a sarcastic response in a Hindi tone, followed by an actual helpful answer.")

    # Load the model and tokenizer
    model, tokenizer = load_model()

    # Load the dataset
    df = load_dataset()

    # User input
    user_input = st.text_input("You: ", "I am so stressed about work and coding")

    # Generate response
    if st.button("Get Response"):
        if user_input.strip() == "":
            st.warning("Please enter a question or message!")
        else:
            with st.spinner("Thinking of a sarcastic response..."):
                # Step 1: Try to find a relevant dialogue from the dataset
                relevant_dialogue = find_relevant_dialogue(user_input, df)
                
                if relevant_dialogue:
                    # If a relevant dialogue is found, use it as the sarcastic response
                    st.success(f"Bot (Sarcastic): {relevant_dialogue}")
                else:
                    # If no relevant dialogue is found, generate a funny Hindi-tone response
                    response = funny_hindi_response()
                    st.success(f"Bot (Sarcastic): {response}")

                # Step 2: Provide an actual helpful answer using Wikipedia
                actual_response = fetch_wikipedia_answer(user_input)
                st.info(f"Bot (Helpful): {actual_response}")

# Run the app
if __name__ == "__main__":
    main()

# Footer
st.write("---")
st.markdown('<center><a href="https://www.instagram.com/suraj_nate/" target="_blank" style="color:white;text-decoration:none">&copy; 2025 @suraj_nate All rights reserved.</a></center>', unsafe_allow_html=True)
