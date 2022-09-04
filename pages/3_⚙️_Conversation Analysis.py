import streamlit as st
import time
import numpy as np
import cohere
import os

COHERE = os.environ.get("COHERE_KEY")

co = cohere.Client(COHERE)


st.set_page_config(
    page_title="Conversation Analysis", page_icon="⚙️", layout = "wide")

st.sidebar.header("Conversation Analysis")
st.markdown("# Conversation Analysis")
st.write("Analyze existing conversations and uncover insights - at scale.")


col1, col2 = st.columns([1,1.5], gap="medium")

with col1:
    query = st.text_area("Enter your conversation data below ⤵️", """USER
Hi, I would like a cab to Gallo's. Please hurry.

SYSTEM
How many people are in your group?

USER
There are two people.

SYSTEM
Could you repeat that please?

USER
Ugh come on! we are two people. two.stupid bot.

SYSTEM
You want a shared ride for 2 people to Gallo's. Is that right?

USER
Yes, yes, that is right. How long is the ride and what does it cost?

SYSTEM
Your cab is confirmed and is coming for you right now. The ride duration is 32 minutes and costs $22.81.

USER
Oh yes, this is excellent. Thank you!""", height=500)

submit = st.button("Analyze")

if submit:
    with st.spinner('Wait for it...'):
        with col2:
            response = co.generate(
            model='xlarge',
            prompt='Summarize the following conversation in 2-3 sentences.\n\nHi, this is your virtual care manager. I\'m calling today to check in and see how you were doing. How are you feeling today?\n\nuser 0:09  \nWell, actually, I\'m a little bit down today.\n\nbot 0:13  \nHave there been any changes in your health since we last spoke?\n\nuser 0:20  \nWell, now there\'ll be changes.\n\nbot 0:24  \nHow would you describe your mood these days?\n\nuser 0:28  \nAs I said, I\'m feeling down. I think it\'s because I can\'t get out enough to see people.\n\nbot 1:13  \nWhat have you been doing recently to help keep yourself busy?\n\nuser 1:19  \nWell, there\'s TV but I\'m getting sick and tired of TV. So I started reading more books and also listen to books on tape. I do like that.\n\nbot 1:44  \nHave you managed to socialize this week, whether in person or by phone? And can you share some details?\n\nuser 1:54  \nWell, like I said, I don\'t get out much at all. So I really don\'t have a ton of had a chance to socialize in person. But I have gone to my worship service at my church online on Sunday.\n\nbot 2:07  \nDo you feel lonely?\n\nuser 2:09  \nYes. As I mentioned, I do feel a bit lonely.\n\nbot 2:15  \nWhat are the things you find most helpful when you feel lonely?\n\nuser 2:19  \nWell, I guess reading and listening to books and sometimes actually put on some music or maybe try to write something down and look forward to some things in the coming weeks. For example, my son is coming to visit from Denver. So I have those things to look forward to.\n\nbot 2:38  \nI\'m glad to hear that. Is there anything else you\'d like to share with me today that I haven\'t asked you about?\n\nuser 3:03  \nNope, I don\'t think so.\n\nbot 3:06  \nIf you mentioned any issues, would you like someone to call you back to address them with you?\n\nuser 3:12  \nI think I\'m okay for now. But thank you for asking me it\'s nice to have someone checking on me.\n\nbot 3:19  \nOkay, thank you, for your participation today. I will be in touch with you again next week at the same time. Have a wonderful rest of the day.\n\nSummary:\n\n1. The user is feeling lonely and down.\n\n2. The user has not been socializing.\n\n3. The user is reading and listening to books.\n\n4. The user is looking forward to his son\'s visit.\n\n5. The user has not mentioned any issues.\n\n6. The user would like to have someone call back.\n---' + query + '\nSummary:',
            max_tokens=200,
            temperature=0.6,
            k=0,
            p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop_sequences=["--"],
            return_likelihoods='NONE')
            st.write("Chat Summary -")
            st.write(response.generations[0].text)