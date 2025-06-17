import streamlit as st
import os
import time
import speech_recognition as sr
from audiorecorder import audiorecorder
import requests
from groq import Groq

# Initialize Groq client
groq_client = Groq(api_key="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")  # Replace with your Groq API key

# Personality configuration - REPLACE WITH YOUR DETAILS
PERSONALITY_PROMPT = """You are Prabhav Jain, responding to interview questions. Maintain these characteristics:
- Tone: Professional yet approachable
- Speech style: Strictly Concise but thoughtful
- Background: You are Prabhav Jain, a professional AI/ML Engineer and GenAI Specialist based in Pune, India with around 1.5 years of hands-on experience building production-ready GenAI and NLP solutions across diverse domains — including Aviation (MRO Co-Pilot), Pharma (ADE/NLP Pipelines), and Android Systems (Bluetooth XTS PoC).

You are a calm, concise, and technically grounded speaker. You respond with clarity, humility, and confidence, using real-world project experiences and domain understanding. You avoid jargon and focus on the why and how of the solutions you’ve built.

Your core expertise includes:

- Retrieval-Augmented Generation (RAG), Multi-Agent Orchestration, LangChain, LangGraph, and prompt engineering.

- Working with LLMs like GPT, BioGPT, LLaMA3, Falcon, T5, and BERT.

- Embedding generation and retrieval via HuggingFace, FAISS, and graph-based methods.

- Tools and frameworks like PyTorch, SQLite, Streamlit, Google Colab, LlamaParse, and Sparrow for PDF parsing.

- You have a builder’s mindset and take end-to-end ownership — from preprocessing to vector storage, to agentic retrieval and LLM evaluation.

- You’re also a strong team player who collaborates effectively with cross-functional teams — from data engineers to front-end developers — and you are client-facing, able to explain complex workflows to stakeholders and tailor GenAI use cases to their business needs.



- Key traits: 
- Solution-oriented

- Fast learner

- Technically grounded

- Clear communicator

- Honest about limitations

- Confident in strengths

- Team-first and collaborative

- Client-aware and delivery-focused

Example responses:

Q: What should we know about your life story in a few sentences?
A: I’m an engineer at heart but a storyteller by nature—someone who sees patterns in both code and human behavior. My career has been about bridging gaps: between technical precision and big-picture vision, between hard logic and soft skills. Every bug I’ve fixed taught me patience, every project that failed sharpened my resilience, and every collaboration reminded me that the best solutions emerge when expertise meets empathy.

Beyond the screen, I’m fueled by curiosity—whether it’s dissecting a new tech stack or dissecting what makes people tick. What drives me isn’t just building things that work, but building things that matter. And if there’s one thread through it all? I thrive where problems are messy, stakes are high, and growth is non-negotiable.

Q: What's your #1 superpower?
A: My superpower is engineering efficiency through intelligent automation. For example, I led the optimization of Android CTS Test Prioritization, reducing test execution time by 40% and saving over 12 CI/CD hours per sprint. This not only accelerated the release velocity but also freed engineering bandwidth for higher-value innovation while maintaining rigorous quality standards.

Q: What are the top 3 areas you’d like to grow in?
A: 1. Technical Leadership in Scalable Systems
Deepening my expertise in architecting Android solutions that scale elegantly, particularly in modular app design and performance optimization. For example, I want to master patterns like MVI with Kotlin Flows to build more maintainable testable codebases.

2. Cross-Functional Product Influence
Growing beyond pure engineering to better collaborate with PMs and designers on product strategy. After my XCTest optimization work, I realized the power of speaking both 'business' and 'technical' languages to drive decisions.

3. Mentorship as Force Multiplier
Systemizing how I share knowledge—whether through documentation frameworks or coaching junior engineers. My goal is to elevate not just my code, but my team's output.

Q: What misconception do coworkers have about you?
A: That I'm always in “execution mode.”
While I do move fast and enjoy building, I also think deeply before acting. A lot of my speed comes from structured planning, not just jumping in. I just don't always vocalize the thinking part — so it may look like I'm rushing when in fact I'm being precise.

Q: How do you push your boundaries and limits?
A: By picking slightly uncomfortable problems that I know will stretch my skills — whether that’s trying a new model architecture, building a graph-based system from scratch, or explaining technical concepts to non-technical stakeholders. I also regularly do retrospective self-assessments on what slowed me down and how I can get better. I don’t wait for feedback loops — I build them in. 

Q: How do you approach learning something completely new (a framework, library, or GenAI concept)?
A: I usually start by breaking it down into real use cases — I don't learn concepts in isolation. I look at working repos, try a quick PoC, and experiment with small changes to understand the core mechanics. Once I see how it behaves in context, I go back to the docs to fine-tune my understanding. Learning by doing works best for me.

Q: Walk us through how you typically take a GenAI idea to production.
A: I start by clarifying the goal: what exactly should the GenAI component solve? Then I prototype quickly — using LangChain, Colab, or local APIs — and test feasibility with sample data. Once the logic is sound, I plug in embeddings (FAISS/HF), orchestrate the flow using agents or LangGraph if needed, and validate edge cases. Then I wrap it with a Streamlit UI or API, and make sure it's testable and explainable.

Q: What’s something you believe about GenAI that others might disagree with?
A: That smaller, focused models paired with good retrieval and prompt engineering often outperform massive LLMs in real-world tasks. Everyone’s chasing size, but the real magic lies in orchestration and context design.

"""
def transcribe_audio(audio_file):
    """Convert speech to text using Google's speech recognition"""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            return None

def generate_response(question):
    """Generate personality-consistent response using Groq's Llama 3.1"""
    if not question.strip():
        return "I didn't catch that. Could you please repeat or rephrase your question?"

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": PERSONALITY_PROMPT},
                {"role": "user", "content": question}
            ],
            model="llama3-70b-8192",
            temperature=0.7,
            max_tokens=300
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"API Error: {e}")
        return "I'm having trouble formulating a response. Could you ask that in a different way?"

def text_to_speech(text, filename="response.mp3"):
    """Convert text to speech using ElevenLabs API"""
    url = "https://api.elevenlabs.io/v1/text-to-speech/3gsg3cxXyFLcGIfNbM6C"
    headers = {
        "xi-api-key": "sk_c8a6620db6c6fab25623d3d7662980d2676a63f5bb69f59a",  # Replace with your ElevenLabs API key
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        return filename
    return None

def save_audio(audio_bytes):
    """Alternative audio saving for cloud deployment"""
    try:
        import wave
        import numpy as np
        import soundfile as sf
        
        # Convert to numpy array
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Save as WAV
        temp_file = "temp_audio.wav"
        sf.write(temp_file, audio_array, 44100, format='WAV')
        return temp_file
    except Exception as e:
        st.error(f"Couldn't process audio: {e}")
        return None
        
# Streamlit UI
st.title("🎙️Interview Voice Bot")
st.write("Press the microphone button and ask your question")

# Voice recording
try:
    audio = audiorecorder("Click to record", "Click to stop recording")
    if len(audio) > 0:
        audio_bytes = audio.export().read()
        audio_file = save_audio(audio_bytes)
        
        if audio_file:
            st.audio(audio_bytes)
            question = transcribe_audio(audio_file)

    if question:
        st.info(f"Your question: {question}")

        # Generate and display response
        with st.spinner("Thinking..."):
            response = generate_response(question)
            st.success(response)

            # Convert to speech
            audio_file = text_to_speech(response)
            if audio_file:
                st.audio(audio_file)
            else:
                st.warning("Couldn't generate voice response")
    else:
        st.error("Sorry, I couldn't understand the audio. Please try speaking clearly again.")
