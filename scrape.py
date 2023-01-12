import pdfkit
import streamlit as st
import re
import PyPDF2
import pandas as pd

#set website link a name
st.set_page_config(page_title="Tobibui1904",layout="wide")

#outlines of the website
header=st.container()
converter = st.container()
searching = st.container()

#Introduction about the program
with header:
    st.markdown("<h1 style='text-align: center; color: lightblue;'>Searcher Optimization Tool</h1>", unsafe_allow_html=True)
    st.caption("<h1 style='text-align: center;'>By Tobi Bui</h1>",unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: left; color: red;'>Introduction about the project</h1>", unsafe_allow_html=True)
    st.subheader('1: Project Purposes')
    st.markdown("""The objective of my project is to make Ctrl F more convenient and funnier. For those who have to deal with a large data of PDFs ore many webpages whose information needed to be found easily,
                my program is for you. It's like a summary version of Ctrl F into a table. With this, you can look for information easily without scrolling down to multiple places and process with various sentences with unnecsary information.
                This table also filters out number appearances if you're looking for a statistics for your research or project. I believe this tool would play an important role to help your life easier. 
""")
    st.subheader('2: How it works')
    st.markdown("""- Step 1: If you have your own PDF files, upload it to the program. One note is that you can upload multiple files. Otherwise, if you want to search for something online, paste the link to the input bar. """)
    st.markdown("""- Step 2: And then put your search term to the bar.""")
    st.markdown("""- Step 3: Waiting for the results.""")
    st.subheader('3: Features')
    st.markdown("""- Display total number of pages after receiving the correct input""")
    st.markdown("""- Display the result with sentences containing them as well as the numerical values in the sentences so that users can extract stats for research or analysis""")
    st.markdown("""- Display the word's time appearance and number of pages it occurs """)
    st.markdown("""- Check legit website links""")
    st.markdown("""- Allow multiple local files to be uploaded""")
    
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
    
    
    
    
    
    

