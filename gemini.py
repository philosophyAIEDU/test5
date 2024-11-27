import streamlit as st
import google.generativeai as genai
import PyPDF2
import io

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def initialize_gemini(api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    return model

def get_team_analysis(text, role):
    prompts = {
        "sam": f"""당신은 Sam이라는 AI PhD 연구원입니다. 
        다음 논문을 주의 깊게 읽고 핵심 포인트, 방법론, 결과를 파악하여 
        간단한 용어로 초안을 작성해주세요:\n{text}""",
        
        "jenny": f"""당신은 AI와 교육 분야의 PhD를 가진 Jenny입니다.
        다음 분석을 검토하고, 더 쉬운 언어로 설명하며, 교육적 맥락과 실제 응용 사례를 추가해주세요:\n{text}""",
        
        "will": f"""당신은 팀 리더 Will입니다. 
        다음 내용을 검토하여 최종 보고서를 작성해주세요. 
        모든 핵심 내용이 포함되어 있는지 확인하고, 
        일관된 톤과 스타일을 유지하며, 
        읽기 쉽게 구조화해주세요:\n{text}"""
    }
    return prompts.get(role, "")

def generate_structured_report(model, text):
    # Sam의 초기 분석
    sam_prompt = get_team_analysis(text, "sam")
    sam_analysis = model.generate_content(sam_prompt).text
    
    # Jenny의 검토 및 개선
    jenny_prompt = get_team_analysis(sam_analysis, "jenny")
    jenny_analysis = model.generate_content(jenny_prompt).text
    
    # Will의 최종 검토
    will_prompt = get_team_analysis(jenny_analysis, "will")
    final_report = model.generate_content(will_prompt).text
    
    return final_report

st.title("AI 논문 분석 챗봇 🤖")

# API 키 입력
api_key = st.text_input("Google API 키를 입력하세요:", type="password")

# PDF 파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")

if uploaded_file and api_key:
    try:
        # PDF 텍스트 추출
        pdf_text = extract_text_from_pdf(uploaded_file)
        
        # Gemini 모델 초기화
        model = initialize_gemini(api_key)
        
        # 분석 옵션
        analysis_option = st.selectbox(
            "원하시는 분석을 선택하세요:",
            ["팀 리뷰 보고서", "논문 요약", "주요 연구 방법론", "연구 결과 분석", "사용자 질문"]
        )
        
        if st.button("분석 시작"):
            with st.spinner("분석 중..."):
                if analysis_option == "팀 리뷰 보고서":
                    st.subheader("🔍 AI 연구팀의 체계적 분석")
                    final_report = generate_structured_report(model, pdf_text)
                    
                    # 보고서를 섹션별로 표시
                    sections = final_report.split("\n\n")
                    for section in sections:
                        if section.strip():
                            st.markdown(section)
                            st.divider()
                
                elif analysis_option == "논문 요약":
                    prompt = f"다음 논문을 500자 이내로 요약해주세요:\n{pdf_text}"
                    response = model.generate_content(prompt)
                    st.write(response.text)
                    
                elif analysis_option == "주요 연구 방법론":
                    prompt = f"다음 논문의 주요 연구 방법론을 설명해주세요:\n{pdf_text}"
                    response = model.generate_content(prompt)
                    st.write(response.text)
                    
                elif analysis_option == "연구 결과 분석":
                    prompt = f"다음 논문의 주요 연구 결과를 분석해주세요:\n{pdf_text}"
                    response = model.generate_content(prompt)
                    st.write(response.text)
                    
                else:
                    user_question = st.text_input("논문에 대해 궁금한 점을 입력하세요:")
                    prompt = f"다음 논문에 대한 질문에 답변해주세요. 질문: {user_question}\n논문 내용:\n{pdf_text}"
                    response = model.generate_content(prompt)
                    st.write(response.text)
                
    except Exception as e:
        st.error(f"오류가 발생했습니다: {str(e)}")
else:
    st.info("API 키를 입력하고 PDF 파일을 업로드해주세요.")
