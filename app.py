import streamlit as st
import ollama

from rag import (
    extract_pdf_text,
    split_text,
    create_embeddings,
    find_relevant_chunks,
    create_rag_prompt
)

from crop_recommendation import (
    train_crop_model,
    recommend_crop
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Smart Farming Assistant",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #f4fff4 0%,
            #eaffea 50%,
            #f8fff8 100%
        );
    }

    /* Main content */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #0f5132 0%,
            #198754 100%
        );
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Sidebar title */
    .sidebar-title {
        text-align: center;
        font-size: 25px;
        font-weight: bold;
        padding: 10px;
        margin-bottom: 15px;
    }

    .sidebar-subtitle {
        text-align: center;
        font-size: 14px;
        margin-bottom: 20px;
    }

    /* Main title */
    .main-title {
        text-align: center;
        color: #146c43;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .main-subtitle {
        text-align: center;
        color: #4f6354;
        font-size: 18px;
        margin-bottom: 30px;
    }

    /* Feature cards */
    .feature-card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #d7ead9;
        box-shadow: 0 5px 18px rgba(0, 80, 30, 0.08);
        margin-bottom: 20px;
    }

    .feature-card h3 {
        color: #146c43;
        margin-bottom: 10px;
    }

    .feature-card p {
        color: #52665a;
        font-size: 15px;
    }

    /* Section heading */
    .section-title {
        color: #146c43;
        font-size: 28px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(
            90deg,
            #198754,
            #20a464
        );
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 22px;
        font-weight: 600;
        transition: 0.3s;
    }

    .stButton > button:hover {
        background: linear-gradient(
            90deg,
            #146c43,
            #198754
        );
        color: white;
        transform: scale(1.02);
    }

    /* Input boxes */
    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #b8d8be;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: white;
        border-radius: 15px;
        padding: 15px;
        border: 1px solid #d7ead9;
        box-shadow: 0 4px 12px rgba(0, 80, 30, 0.06);
    }

    /* Recommendation box */
    .recommendation-box {
        background: linear-gradient(
            135deg,
            #d8f3dc,
            #f1fff3
        );
        border-left: 6px solid #198754;
        border-radius: 15px;
        padding: 25px;
        margin-top: 20px;
        text-align: center;
    }

    .recommendation-title {
        color: #146c43;
        font-size: 18px;
        font-weight: 600;
    }

    .recommendation-crop {
        color: #0b5e32;
        font-size: 35px;
        font-weight: 800;
        margin-top: 8px;
    }

    /* Info boxes */
    .info-box {
        background: #ffffff;
        border-radius: 15px;
        padding: 20px;
        border: 1px solid #d7ead9;
        margin-bottom: 20px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #63756a;
        font-size: 13px;
        padding: 25px;
        margin-top: 40px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌱 AI Smart Farming Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'AI-powered assistance for smarter and more informed farming decisions'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🌾 Smart Farmer</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'AI Agricultural Assistant'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### 📌 Select a Feature")

    option = st.radio(
        "Choose one:",
        [
            "🏠 Home",
            "🤖 AI Farming Chat",
            "📄 Ask Questions from PDF",
            "🌱 Crop Recommendation",
            "💧 Smart Irrigation",
            "🐛 Pest & Disease Guidance",
            "🌾 Farming Tips",
            "ℹ️ About Project"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### 🌱 Project Features")

    st.markdown(
        """
        **🤖 AI Chat**  
        Ask farming questions.

        **📄 PDF Assistant**  
        Ask questions from agricultural documents.

        **🌱 Crop Recommendation**  
        Predict a suitable crop.

        **💧 Smart Irrigation**  
        Get basic irrigation guidance.

        **🐛 Pest & Disease**  
        Get general crop-care guidance.

        **🌾 Farming Tips**  
        Useful agricultural practices.
        """
    )

    st.divider()

    st.caption("AI Smart Farming Assistant")
    st.caption("Built with Python + Streamlit + AI")


# ============================================================
# OLLAMA FUNCTION
# ============================================================

def ask_ollama(prompt):

    try:

        response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "system",
                    "content": """
You are an AI Smart Farming Assistant.

Help farmers with:
- crops
- soil
- irrigation
- fertilizers
- pests
- agricultural practices
- crop management
- farming problems

Give simple, clear and easy-to-understand answers.

Use headings and bullet points when useful.

Do not invent information.

For serious crop disease, pesticide or chemical decisions,
recommend consulting a qualified agricultural expert.
"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]

    except Exception as e:

        return (
            "Unable to connect to Ollama.\n\n"
            "Please make sure Ollama is installed and running.\n\n"
            f"Error: {e}"
        )


# ============================================================
# HOME
# ============================================================

if option == "🏠 Home":

    st.markdown(
        '<div class="section-title">🏠 Welcome to AI Smart Farming Assistant</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
        <h3>🌾 Smart Farming with Artificial Intelligence</h3>
        <p>
        This application helps farmers use Artificial Intelligence
        to understand agricultural information, ask farming questions,
        analyze agricultural documents and get crop recommendations.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="feature-card">
                <h3>🤖 AI Farming Chat</h3>
                <p>
                Ask questions about crops, soil, irrigation,
                fertilizers and general farming.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="feature-card">
                <h3>📄 PDF Assistant</h3>
                <p>
                Upload an agricultural PDF and ask questions
                using AI-powered document retrieval.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="feature-card">
                <h3>🌱 Crop Recommendation</h3>
                <p>
                Enter soil and environmental values to
                predict a suitable crop.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    col4, col5, col6 = st.columns(3)

    with col4:

        st.markdown(
            """
            <div class="feature-card">
                <h3>💧 Smart Irrigation</h3>
                <p>
                Get basic irrigation suggestions based on
                crop and environmental conditions.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col5:

        st.markdown(
            """
            <div class="feature-card">
                <h3>🐛 Pest & Disease</h3>
                <p>
                Ask AI about common crop pest and disease
                symptoms and general prevention practices.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col6:

        st.markdown(
            """
            <div class="feature-card">
                <h3>🌾 Farming Tips</h3>
                <p>
                Learn useful agricultural practices for
                better crop management.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# AI FARMING CHAT
# ============================================================

elif option == "🤖 AI Farming Chat":

    st.markdown(
        '<div class="section-title">🤖 AI Farming Chat</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
        <b>👨‍🌾 Ask your farming question</b><br>
        You can ask about crops, soil, irrigation,
        fertilizers, pests and agricultural practices.
        </div>
        """,
        unsafe_allow_html=True
    )

    question = st.text_area(
        "Enter your farming question:",
        placeholder="Example: What is the best irrigation method for rice?",
        height=120
    )

    if st.button(
        "🌱 Ask AI",
        type="primary"
    ):

        if question.strip():

            with st.spinner("🤖 AI is thinking..."):

                answer = ask_ollama(question)

            st.subheader("🌾 AI Answer")

            st.write(answer)

        else:

            st.warning(
                "Please enter a farming question."
            )


# ============================================================
# PDF QUESTION ANSWERING
# ============================================================

elif option == "📄 Ask Questions from PDF":

    st.markdown(
        '<div class="section-title">📄 Agricultural PDF Assistant</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
        Upload an agricultural document and ask questions
        based on the information available inside the PDF.
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "📂 Upload your agricultural PDF",
        type=["pdf"]
    )

    if uploaded_file:

        with st.spinner(
            "📖 Reading your agricultural document..."
        ):

            pdf_text = extract_pdf_text(
                uploaded_file
            )

        if not pdf_text.strip():

            st.error(
                "Could not extract text from this PDF. "
                "Please try a text-based PDF."
            )

        else:

            st.success(
                "✅ PDF successfully processed!"
            )

            st.info(
                f"📄 Extracted approximately "
                f"{len(pdf_text)} characters."
            )

            chunks = split_text(
                pdf_text,
                chunk_size=500
            )

            st.info(
                f"🧩 Created {len(chunks)} text chunks."
            )

            with st.spinner(
                "🧠 Creating document embeddings..."
            ):

                embeddings = create_embeddings(
                    chunks
                )

            question = st.text_area(
                "👨‍🌾 Ask a question about your PDF:",
                placeholder="Example: What fertilizer is recommended in this document?",
                height=100
            )

            if st.button(
                "🔍 Search PDF and Ask AI",
                type="primary"
            ):

                if question.strip():

                    with st.spinner(
                        "🔎 Finding relevant information..."
                    ):

                        relevant_chunks = (
                            find_relevant_chunks(
                                question,
                                chunks,
                                embeddings,
                                top_k=3
                            )
                        )

                    if relevant_chunks:

                        prompt = create_rag_prompt(
                            question,
                            relevant_chunks
                        )

                        with st.spinner(
                            "🤖 Generating answer..."
                        ):

                            answer = ask_ollama(
                                prompt
                            )

                        st.subheader(
                            "🌾 AI Answer"
                        )

                        st.write(answer)

                        with st.expander(
                            "📚 View Retrieved Information"
                        ):

                            for i, chunk in enumerate(
                                relevant_chunks,
                                start=1
                            ):

                                st.markdown(
                                    f"**Relevant Section {i}**"
                                )

                                st.write(chunk)

                                st.divider()

                    else:

                        st.warning(
                            "No relevant information found."
                        )

                else:

                    st.warning(
                        "Please enter a question."
                    )


# ============================================================
# CROP RECOMMENDATION
# ============================================================

elif option == "🌱 Crop Recommendation":

    st.markdown(
        '<div class="section-title">🌱 Crop Recommendation System</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
        Enter the soil nutrients and environmental conditions.
        The Machine Learning model will predict a suitable crop.
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.spinner(
        "🌾 Preparing crop recommendation model..."
    ):

        model, label_encoder, result = (
            train_crop_model()
        )

    if model is None:

        st.error(result)

        st.stop()

    else:

        st.success(
            f"✅ Model trained successfully!"
        )

        st.metric(
            "Model Accuracy",
            f"{result * 100:.2f}%"
        )

    st.subheader("🌱 Enter Crop Conditions")

    col1, col2, col3 = st.columns(3)

    with col1:

        nitrogen = st.number_input(
            "Nitrogen (N)",
            min_value=0.0,
            max_value=200.0,
            value=50.0,
            step=1.0
        )

        phosphorus = st.number_input(
            "Phosphorus (P)",
            min_value=0.0,
            max_value=200.0,
            value=50.0,
            step=1.0
        )

        potassium = st.number_input(
            "Potassium (K)",
            min_value=0.0,
            max_value=250.0,
            value=50.0,
            step=1.0
        )

    with col2:

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=0.0,
            max_value=60.0,
            value=25.0,
            step=0.1
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=0.1
        )

    with col3:

        ph = st.number_input(
            "Soil pH",
            min_value=0.0,
            max_value=14.0,
            value=6.5,
            step=0.1
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=500.0,
            value=100.0,
            step=1.0
        )

    st.write("")

    if st.button(
        "🌾 Recommend Crop",
        type="primary"
    ):

        crop = recommend_crop(
            model,
            label_encoder,
            nitrogen,
            phosphorus,
            potassium,
            temperature,
            humidity,
            ph,
            rainfall
        )

        st.markdown(
            f"""
            <div class="recommendation-box">
                <div class="recommendation-title">
                    🌱 Recommended Crop
                </div>
                <div class="recommendation-crop">
                    {crop.upper()}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info(
            "The recommendation is based on the soil and "
            "environmental values entered above."
        )


# ============================================================
# SMART IRRIGATION
# ============================================================

elif option == "💧 Smart Irrigation":

    st.markdown(
        '<div class="section-title">💧 Smart Irrigation Assistant</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
        Enter the crop and current environmental conditions.
        AI will provide general irrigation guidance.
        </div>
        """,
        unsafe_allow_html=True
    )

    crop_name = st.text_input(
        "🌱 Crop Name",
        placeholder="Example: Rice"
    )

    soil_moisture = st.number_input(
        "💧 Soil Moisture (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

    temperature = st.number_input(
        "🌡️ Temperature (°C)",
        min_value=0.0,
        max_value=60.0,
        value=25.0
    )

    humidity = st.number_input(
        "☁️ Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=60.0
    )

    rainfall = st.number_input(
        "🌧️ Recent Rainfall (mm)",
        min_value=0.0,
        max_value=500.0,
        value=10.0
    )

    if st.button(
        "💧 Get Irrigation Advice",
        type="primary"
    ):

        if crop_name.strip():

            irrigation_prompt = f"""
Give simple general irrigation guidance for a farmer.

Crop: {crop_name}
Soil moisture: {soil_moisture}%
Temperature: {temperature}°C
Humidity: {humidity}%
Recent rainfall: {rainfall} mm

Explain:
1. Whether irrigation may currently be needed.
2. What factors should be checked before watering.
3. General precautions.

Do not give dangerous or highly specific chemical advice.
"""

            with st.spinner(
                "💧 Preparing irrigation advice..."
            ):

                answer = ask_ollama(
                    irrigation_prompt
                )

            st.subheader("🌾 Irrigation Guidance")

            st.write(answer)

        else:

            st.warning(
                "Please enter the crop name."
            )


# ============================================================
# PEST AND DISEASE GUIDANCE
# ============================================================

elif option == "🐛 Pest & Disease Guidance":

    st.markdown(
        '<div class="section-title">🐛 Pest & Disease Guidance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
        Describe the symptoms you see on the crop.
        AI will provide general information and possible
        causes to discuss with an agricultural expert.
        </div>
        """,
        unsafe_allow_html=True
    )

    crop_name = st.text_input(
        "🌱 Crop Name",
        placeholder="Example: Tomato"
    )

    symptoms = st.text_area(
        "🔍 Describe the symptoms",
        placeholder=(
            "Example: Leaves are turning yellow and "
            "small spots are appearing."
        ),
        height=150
    )

    if st.button(
        "🐛 Analyze Symptoms",
        type="primary"
    ):

        if crop_name.strip() and symptoms.strip():

            disease_prompt = f"""
You are helping a farmer understand a possible crop problem.

Crop: {crop_name}

Symptoms:
{symptoms}

Give:
1. Possible general causes.
2. What the farmer should observe next.
3. General prevention and crop-care practices.
4. When to consult a qualified agricultural expert.

Do not provide unsafe pesticide or chemical instructions.
Do not claim a diagnosis with certainty.
"""

            with st.spinner(
                "🔎 Analyzing crop symptoms..."
            ):

                answer = ask_ollama(
                    disease_prompt
                )

            st.subheader(
                "🌿 AI Crop-Care Guidance"
            )

            st.write(answer)

        else:

            st.warning(
                "Please enter both crop name and symptoms."
            )


# ============================================================
# FARMING TIPS
# ============================================================

elif option == "🌾 Farming Tips":

    st.markdown(
        '<div class="section-title">🌾 Farming Tips</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
        Select a topic to receive simple AI-generated
        agricultural guidance.
        </div>
        """,
        unsafe_allow_html=True
    )

    topic = st.selectbox(
        "Choose a farming topic:",
        [
            "Soil Preparation",
            "Seed Selection",
            "Fertilizer Management",
            "Water Management",
            "Crop Management",
            "Pest Prevention",
            "Harvesting",
            "Post-Harvest Management"
        ]
    )

    if st.button(
        "🌾 Get Farming Tips",
        type="primary"
    ):

        tips_prompt = f"""
Give simple and useful farming tips about:

{topic}

Give the answer in:
- 5 important points
- simple language
- practical general guidance

Avoid unsafe chemical instructions.
"""

        with st.spinner(
            "🌱 Preparing farming tips..."
        ):

            answer = ask_ollama(
                tips_prompt
            )

        st.subheader(
            f"🌾 {topic} Tips"
        )

        st.write(answer)


# ============================================================
# ABOUT PROJECT
# ============================================================

elif option == "ℹ️ About Project":

    st.markdown(
        '<div class="section-title">ℹ️ About the Project</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">

        <h3>🌱 AI Smart Farming Assistant</h3>

        <p>
        AI Smart Farming Assistant is an Artificial Intelligence
        based agricultural support system designed to provide
        farmers with useful information through a simple interface.
        </p>

        <h4>🚀 Main Technologies</h4>

        <ul>
            <li>🐍 Python</li>
            <li>🎨 Streamlit</li>
            <li>🤖 Ollama + Llama 3.2</li>
            <li>🧠 Sentence Transformers</li>
            <li>📄 PyPDF</li>
            <li>📊 Scikit-learn</li>
            <li>🌳 Random Forest</li>
            <li>🔎 Retrieval Augmented Generation (RAG)</li>
        </ul>

        <h4>✨ Main Features</h4>

        <ul>
            <li>AI Farming Chat</li>
            <li>Agricultural PDF Question Answering</li>
            <li>Machine Learning Crop Recommendation</li>
            <li>Smart Irrigation Guidance</li>
            <li>Pest and Disease Guidance</li>
            <li>AI Farming Tips</li>
        </ul>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🌱 <b>AI Smart Farming Assistant</b><br>
        Built using Python, Streamlit, Ollama,
        Machine Learning and RAG
    </div>
    """,
    unsafe_allow_html=True
)
