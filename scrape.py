import pdfkit
import streamlit as st
import re
import PyPDF2
import pandas as pd
import numpy as np
import altair as alt
import uuid

#NLP Sentiment Analysis library
import neattext.functions as nfx
import joblib

#NLP Translation Library
from mtranslate import translate
import os
from gtts import gTTS
import base64

# ML libraries
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

#set website link a name
st.set_page_config(page_title="Tobibui1904",layout="wide")

#outlines of the website
header=st.container()
converter = st.container()
searching = st.container()
comment = st.container()

#Introduction about the program
with header:
    st.markdown("<h1 style='text-align: center; color: lightblue;'>Searcher Optimization Tool</h1>", unsafe_allow_html=True)
    st.caption("<h1 style='text-align: center;'>By Tobi Bui</h1>",unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: left; color: red;'>Introduction about the project</h1>", unsafe_allow_html=True)
    st.subheader('1: Project Purposes')
    st.markdown("""The objective of my project is to make Ctrl F more convenient and funnier. For those who have to deal with a large data of PDFs ore many webpages whose information needed to be found easily,
                my program is for you. It's like a summary version of Ctrl F into a table. With this, you can look for information easily without scrolling down to multiple places and process with various sentences with unnecsary information.
                This table also filters out number appearances if you're looking for a statistics for your research or project. Finally, you can leave your comment at the bottom of the program, and let my NLP application handle it. I believe this tool would play an important role to help your life easier. 
""")
    st.subheader('2: How it works')
    st.markdown("""- Step 1: If you have your own PDF files, upload it to the program. One note is that you can upload multiple files. Otherwise, if you want to search for something online, paste the link to the input bar. """)
    st.markdown("""- Step 2: And then put your search term to the bar.""")
    st.markdown("""- Step 3: Waiting for the results.""")
    st.markdown("""- Step 4 (Optional): You can leave your comment below and witness the magic of NLP.""")
    st.subheader('3: Features')
    st.markdown("""- Display total number of pages after receiving the correct input""")
    st.markdown("""- Display the result with sentences containing them as well as the numerical values in the sentences so that users can extract stats for research or analysis""")
    st.markdown("""- Display the word's time appearance and number of pages it occurs """)
    st.markdown("""- Check legit website links""")
    st.markdown("""- Allow multiple local files to be uploaded""")
    st.markdown("""- NLP Implementation to your comment to satisfy the client""")
    
    st.write("---")

with converter:
    st.markdown("<h1 style='text-align: left; color: red;'>Converter</h1>", unsafe_allow_html=True)
    #configuration for wkhtmltopdf
    path_wkhtmltopdf = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
    config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

    #Input webpages
    uploaded_file = st.text_input('Website link to search')
    
    #test whether it's a legit link
    regex = r'^https?://\S+$'
    def is_http_link(link):
        return bool(re.match(regex, link))

    #if this's a good link, make it downloadable
    if is_http_link(uploaded_file) == True or is_http_link(uploaded_file) is None:
        pdfkit.from_url(uploaded_file, "out.pdf" ,configuration=config)

        with open("out.pdf", "rb") as pdf_file:
            PDFbyte = pdf_file.read()

        pdf = st.download_button(label="PDF File",
                            data=PDFbyte,
                            file_name="Download.pdf",
                            mime='application/octet-stream')

    #otherwise, upload from local 
    else:
        st.write("Not the correct link. You can either paste a new link or upload pdf from your local to the below section.")
    
    #Local PDF upload from computer
    local_pdf = st.file_uploader('Please choose your .pdf file', type="pdf", accept_multiple_files= True)
    
    st.write("---")

with searching:
    st.markdown("<h1 style='text-align: left; color: red;'>Functions</h1>", unsafe_allow_html=True)
    
    #Assign file to doc variable
    if local_pdf == []:
        doc = PyPDF2.PdfReader("out.pdf")
    else:
        #Handle single PDF file
        if 1<= len(local_pdf) < 2:
            doc = PyPDF2.PdfReader(local_pdf[0])
        
        #Handle multiple PDF files and merge them into 1
        else:
            merger = PyPDF2.PdfMerger()
            for pdf in local_pdf:
                pdf_reader = PyPDF2.PdfReader(pdf)
                merger.append(pdf_reader)
            merger.write("merged.pdf")
            doc = merger
        
    #Number of pages:
    st.subheader("Total number of pages:" + " " + str(len(doc.pages)))
    
    # Find integer appearances in a string
    def find_integer(sentence):
        integers = [int(x) for x in sentence.split() if x.isdigit()]
        return integers
        
    # Ctrl F Implementation using regex
    def find_sentence(text, search_term):
        matches = [match for match in re.finditer(search_term, text)]
        list_sentence = []
        list_integer = []
        list_columns = []
        count_sentence = 0
        count_term = 0
        sentences = set()
        if matches:
            for match in matches:
                count_term +=1
                start, end = match.start(), match.end()
                for sentence in re.findall(r'[^.!?]+[.!?]', text):
                    if text.find(sentence) <= start < end <= text.find(sentence) + len(sentence) and sentence not in sentences:
                        sentences.add(sentence)
                        count_sentence +=1
                        list_sentence.append(sentence)
                        list_integer.append(find_integer(sentence))
                        break
            
            #Create dataframe from the matching list
            df = pd.DataFrame(list_integer)
            df['Number'] = df[df.columns[1:]].apply(
                lambda x: ','.join(x.dropna().astype(str)),
                axis=1
            )
            
            # Put the names of columns to a list
            for col in df.columns:
                list_columns.append(col)
            
            # Add the Sentence column to the dataframe
            df["Sentence"] = list_sentence
            
            # Replace empty cell with None
            df['Number'] = df['Number'].astype(str).replace("", 'None')
            
            # Change the position of the 2 columns
            df1 = df.iloc[:,[len(list_columns), len(list_columns) - 1]]
            
            #Print out the dataframe
            st.dataframe(df1,6000,1000)
            
            #Print statistics about the word    
            st.write("The word appears " + str(count_term) + " times" + " in " + str(count_sentence) + " sentences")    
        else:
            st.write("Word not found.")
        
    # Combine all text convert from every pages into one
    #If this is a single PDF file
    if type(doc) != PyPDF2.PdfMerger:
        combined =''
        for i in range(len(doc.pages)):
            current_page = doc.pages[i]
            text = current_page.extract_text()
            text = text.lower()
            combined += text

    #If this is a multiple PDF files merger
    else:
        pdf_reader = PyPDF2.PdfReader(open("merged.pdf", 'rb'))
        combined = ""
        for page_num in range( len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            text = text.lower()
            combined += text
        
    # Store the initial value of widgets in session state
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

    col1, col2 = st.columns(2)

    with col1:
        #Disable the input search feature
        st.checkbox("Disable text input widget", key="disabled")

    with col2:
        #Input of the search term
        search = st.text_input(
            "Enter some word 👇",
            disabled=st.session_state.disabled
        )

        # Call out the find_sentence() function above
        if search:
            search = search.lower()
            find_sentence(combined, search)
    
    st.write("---")

with comment:
    st.markdown("<h1 style='text-align: left; color: red;'>Comment</h1>", unsafe_allow_html=True)
    
    # Load language dataset
    df2 = pd.read_csv(r"C:\Users\Admin\Scraping\language.csv", encoding= 'unicode_escape')
    
    # Language Data Cleaning
    df2.dropna(inplace=True)
    lang = df2['name'].to_list()
    langlist=tuple(lang)
    langcode = df2['iso'].to_list()
    
    choice = st.sidebar.radio('SELECT LANGUAGE',langlist)

    speech_langs = {
        "af": "Afrikaans",
        "ar": "Arabic",
        "bg": "Bulgarian",
        "bn": "Bengali",
        "bs": "Bosnian",
        "ca": "Catalan",
        "cs": "Czech",
        "cy": "Welsh",
        "da": "Danish",
        "de": "German",
        "el": "Greek",
        "en": "English",
        "eo": "Esperanto",
        "es": "Spanish",
        "et": "Estonian",
        "fi": "Finnish",
        "fr": "French",
        "gu": "Gujarati",
        "hi": "Hindi",
        "hr": "Croatian",
        "hu": "Hungarian",
        "hy": "Armenian",
        "id": "Indonesian",
        "is": "Icelandic",
        "it": "Italian",
        "ja": "Japanese",
        "jw": "Javanese",
        "km": "Khmer",
        "kn": "Kannada",
        "ko": "Korean",
        "la": "Latin",
        "lv": "Latvian",
        "mk": "Macedonian",
        "ml": "Malayalam",
        "mr": "Marathi",
        "my": "Myanmar (Burmese)",
        "ne": "Nepali",
        "nl": "Dutch",
        "no": "Norwegian",
        "pl": "Polish",
        "pt": "Portuguese",
        "ro": "Romanian",
        "ru": "Russian",
        "si": "Sinhala",
        "sk": "Slovak",
        "sq": "Albanian",
        "sr": "Serbian",
        "su": "Sundanese",
        "sv": "Swedish",
        "sw": "Swahili",
        "ta": "Tamil",
        "te": "Telugu",
        "th": "Thai",
        "tl": "Filipino",
        "tr": "Turkish",
        "uk": "Ukrainian",
        "ur": "Urdu",
        "vi": "Vietnamese",
        "zh-CN": "Chinese"
    }
    
    # Create dictionary of language and 2 letter langcode
    lang_array = {lang[i]: langcode[i] for i in range(len(langcode))}
    
    # Load emotion datasets
    df1 = pd.read_csv(r"C:\Users\Admin\Scraping\emotion_dataset_2.csv")
    
    # Emotional Data Cleaning
    dir(nfx)
    
    # User Handles
    df1['Clean Text'] = df1['Text'].apply(nfx.remove_userhandles)
    
    # Stopwords
    df1['Clean Text'] = df1['Clean Text'].apply(nfx.remove_stopwords)
    
    # Special Character
    df1['Clean Text'] = df1['Clean Text'].apply(nfx.remove_special_characters)
    
    # Features & Labels
    Xfearures = df1['Clean Text']
    ylabels = df1['Emotion']
    
    # Split Data
    x_train, x_test, y_train, y_test = train_test_split(Xfearures, ylabels, test_size=0.3, random_state=42)
    
    # Logistic Regression Pipeline
    pipe_lr = Pipeline(steps=[("cv", CountVectorizer()), ('lr', LogisticRegression())])
    
    # Train and Fit Data
    pipe_lr.fit(x_train, y_train)
    
    emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}

    # Repeat the input until the user is satisfied then stop
    done = False
    while not done:
        #Input the comment
        exl = st.text_input(
                "Leave your comment below 👇")
        
        emotion = pipe_lr.predict([exl])[0]
        #Emoji representing different emotions
        emoji_icon = emotions_emoji_dict[emotion]
        
        # Make A Prediction
        st.write("From the comment above, I can see that your emotion is " + emotion + emoji_icon)
        
        # Function to decode audio file for download
        def get_binary_file_downloader_html(bin_file, file_label='File'):
            with open(bin_file, 'rb') as f:
                data = f.read()
            bin_str = base64.b64encode(data).decode()
            href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
            return href
        
        # Translation Function
        def translation(c1, c2,exl):
            # Generate random unique key id to avoid the st.text_area error
            key = str(uuid.uuid1())
            
            if len(exl) > 0 :
                try:
                    output = translate(exl,lang_array[choice])
                    with c1:
                        st.markdown("<h1 style='text-align: left; color: red;'>NLP Translation for more application.</h1>", unsafe_allow_html=True)
                        st.text_area("TRANSLATED TEXT",output,height=200, key=key)
                    # if speech support is available will render autio file
                    if choice in speech_langs.values():
                        with c2:
                            aud_file = gTTS(text=output, lang=lang_array[choice], slow=False)
                            aud_file.save("lang.mp3")
                            audio_file_read = open('lang.mp3', 'rb')
                            audio_bytes = audio_file_read.read()
                            bin_str = base64.b64encode(audio_bytes).decode()
                            st.audio(audio_bytes, format='audio/mp3')
                            st.markdown(get_binary_file_downloader_html("lang.mp3", 'Audio File'), unsafe_allow_html=True)
                except Exception as e:
                    st.error(e)
        
        c1, c2 = st.columns(2)
        translation(c1,c2,exl)
        
        st.subheader("Here's some statistics from your comment that my NLP system got:")
        
        # Check Accuracy
        st.success("Accuracy")
        st.write("The accuracy of my NLP estimation of emotion from your comment: " + str(pipe_lr.score(x_test, y_test)))
        
        # Average Prediction Probability
        st.success("Prediction Probability")
        st.write("The average prediction probability of my NLP estimation of emotion from your comment: " + str(np.average(pipe_lr.predict_proba([exl]))))
        
        left1, right1 = st.columns(2)
        
        # Table for different emotion probability
        with left1:
            st.write("Here's the table and graphical visual for different emotion probability:")
            proba_df = pd.DataFrame(pipe_lr.predict_proba([exl]), columns=pipe_lr.classes_)
            
            # Transpose Dataframe
            proba_df = proba_df.T
            
            #Reset column names
            proba_df.columns = ["Probability"]
            st.write(proba_df)
        
        # Bar chart for the emotion and probability table
        with right1:
            proba_df_clean = proba_df.reset_index()
            proba_df_clean.columns = ['Emotions', "Probability"]
            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='Emotions', y = "Probability")
            st.altair_chart(fig, use_container_width = True)
        
        if emotion == "joy" or emotion == 'surprise':
            st.write("It seems my system works well. I hope you enjoy using it for your work.")
            done = True
        else:
            exl = st.text_input(
                "Please tell me what I can do to improve the system 👇")
            
            c3, c4 = st.columns(2)
            translation(c3,c4,exl)
            
            if exl !='':
                st.write('Thank you so much for your feedback. I will try to improve the error ASAP. I wish you a great day and enjoy using my service')
            done = True
    
    
    
    
    
    

