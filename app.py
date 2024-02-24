import streamlit as st
import requests
import torch
from PIL import Image
from io import BytesIO
import torch.nn.functional as F
from matplotlib import cm
import matplotlib.pyplot as plt


st.title("가상피팅 AI 프로젝트")
st.divider()

if "model" not in st.session_state:
    from inference import Inferencer

    device='cuda' if torch.cuda.is_available() else 'cpu'
    st.session_state.model = Inferencer(device=device)

## session state setting
if "human_image" not in st.session_state:
    st.session_state.human_image = None
if "cloth_image" not in st.session_state:
    st.session_state.cloth_image = None
if "new_image" not in st.session_state:
    st.session_state.new_image = None

############################################
###### Step1 : human Image
############################################

st.markdown("#### 자신의 전신 모습이 담긴 이미지를 올려주세요!")
with st.container(border=True):

    col11, col12 = st.columns([0.6,0.4])
    with col11:
        ## select how to load image
        option = st.radio("이미지를 올릴 방법을 선택해주세요.", ("업로드", "URL", "샘플 이미지"), key="option1")
        upload_help = "오른쪽 이미지의 리셋을 위해서는 아래 이미지 파일의 X 버튼을 먼저 눌러주세요."

        ## load image from upload
        if option == "업로드":
            st.markdown("")
            with st.container(border=True):
                file = st.file_uploader("원하시는 이미지를 올려주세요.", type=["png","jpg","jpeg"], help=upload_help, key="upload1")
                if file is not None:
                    image = Image.open(file).convert("RGB")
                    st.session_state.human_image = image
                else:
                    image = None

        ## load image from url
        elif option == "URL":
            with st.form(key="url1"):
                st.markdown("")
                url1 = st.text_input("이미지 주소(URL)를 입력해 주세요.", key="url1")
                button_url = st.form_submit_button("Load")
                if button_url:
                    image1 = Image.open(requests.get(url1, stream=True).raw).convert("RGB")
                    st.session_state.human_image = image1
                else:
                    image1 = None

        elif option == "샘플 이미지":
            st.markdown("")
            st.markdown("")
            st.markdown("")
            with st.container(border=True):
                body_sample_dict = {"select": None,
                                    "사람1": "images/body1.jpg",
                                    "사람2": "images/body2.jpg",
                                    "사람3": "images/body3.jpg"}
                style = st.selectbox("샘플 이미지를 선택해 주세요.", list(body_sample_dict.keys()), key="smaple1")
                if style != "select":
                    image3 = Image.open(body_sample_dict[style]).convert("RGB")
                    st.session_state.human_image = image3
                else:
                    image3 = None

    with col12:
        col121, col122, col123 = st.columns([0.3,0.4,0.3])
        with col122:
            st.markdown("")
            if st.session_state.human_image:
                st.image(st.session_state.human_image, use_column_width=True)
            else:
                no_image = Image.open("images/no_image.png").convert("RGB")
                st.image(no_image, use_column_width=True)
            
            ## "reset" button
            button_reset1 = st.button("Reset", key="reset_step1", use_container_width=True)
            if button_reset1:
                st.session_state.human_image = None
                st.experimental_rerun()  # Re-run the app to reset the state

############################################
###### Step2 : cloth Image
############################################

st.markdown("#### 바꾸고 싶은 옷 이미지를 올려주세요!")
with st.container(border=True):

    col21, col22 = st.columns([0.6,0.4])
    with col21:
        option = st.radio("이미지를 올릴 방법을 선택해주세요.", ("업로드", "URL", "샘플 이미지"), key="option2")
        ## load image from upload
        if option == "업로드":
            st.markdown("")
            with st.container(border=True):
                file2 = st.file_uploader("원하시는 이미지를 올려주세요.", type=["png","jpg","jpeg"], help=upload_help, key="upload2")
                if file2 is not None:
                    image2 = Image.open(file2).convert("RGB")
                    st.session_state.cloth_image = image2
                else:
                    image2 = None

        ## load image from url
        elif option == "URL":
            with st.form(key="url2"):
                st.markdown("")
                url2 = st.text_input("이미지 주소(URL)를 입력해 주세요.", key="url2")
                button_url = st.form_submit_button("Load")
                if button_url:
                    image4 = Image.open(requests.get(url2, stream=True).raw).convert("RGB")
                    st.session_state.cloth_image = image4
                else:
                    image4 = None

        elif option == "샘플 이미지":
            st.markdown("")
            st.markdown("")
            st.markdown("")
            with st.container(border=True):
                cloth_sample_dict = {"select": None,
                                     "긴팔": "images/upper1.jpg",
                                     "반팔": "images/upper2.jpg",
                                     "긴바지": "images/lower1.jpg",
                                     "드레스": "images/dress1.jpg"}
                style = st.selectbox("샘플 이미지를 선택해 주세요.", list(cloth_sample_dict.keys()), key="sample2")
                if style != "select":
                    image = Image.open(cloth_sample_dict[style]).convert("RGB")
                    st.session_state.cloth_image = image
                else:
                    image = None

    with col22:
        col221, col222, col223 = st.columns([0.3,0.4,0.3])
        with col222:
            st.markdown("")
            if st.session_state.cloth_image:
                ## show cloth image
                st.image(st.session_state.cloth_image, use_column_width=True)
            else:
                no_image = Image.open("images/no_image.png").convert("RGB")
                st.image(no_image, use_column_width=True)

            ## "reset" button
            button_reset2 = st.button("Reset", key="reset_step2", use_container_width=True)
            if button_reset2:
                st.session_state.cloth_image = None
                st.experimental_rerun()

############################################
###### Step3 : Fitting image
############################################

st.divider()
st.markdown("#### 가상피팅을 진행해주세요!")

with st.container(border=True):
    col31, col32 = st.columns([0.6,0.4])

    with col31:
        guidance_scale_help = "Guidance scale이 높으면 모델이 옷의 특성을 더 많이 반영하도록 합니다. 하지만 가상피팅된 이미지가 부자연스럽게 보일 수도 있으므로 적당한 값을 선택하는 것이 좋습니다."
        num_inference_steps_help = "Inference step이 높을수록 모델이 더 정교한 이미지를 만듭니다. 하지만 시간이 오래 걸릴 수도 있으므로 적당한 값을 선택하는 것이 좋습니다."

        gscale = st.slider('Guidance scale 값을 선택해주세요.', min_value=0.0, max_value=10.0, value=5.0, help=guidance_scale_help)
        numstep = st.slider('Inference step 수를 선택해주세요.', min_value=1, max_value=50, value=25, help=num_inference_steps_help)
        option = st.radio("바꾸고 싶은 옷의 카테고리를 선택해주세요.",("전신","상의","하의"), horizontal = True)

        if option == "전신":
            category = "dresses"
        elif option == "상의":
            category = "upper_body"
        else:
            category = "lower_body"

    with col32:
        col321, col322, col323 = st.columns([0.3,0.4,0.3])
        with col322:
            st.markdown("")
            if st.session_state.new_image:
                ## show fitting image
                st.image(st.session_state.new_image, use_column_width=True)
            else:
                no_image = Image.open("images/no_image.png").convert("RGB")
                st.image(no_image, use_column_width=True)
            
            ## "fitting" button
            fitting_button = st.button("Fitting", key="fitting", use_container_width=True)
            if fitting_button:
                if st.session_state.human_image is None or  st.session_state.cloth_image is None:
                    st.warning('위 이미지들을 먼저 올려주세요.', icon="⚠️")
                else:
                    fitting_image = st.session_state.model.inference(st.session_state.human_image, st.session_state.cloth_image, category, guidance_scale= gscale, num_inference_steps=numstep)
                    st.session_state.new_image = fitting_image
                    st.experimental_rerun()
            

            # "download" button
            if st.session_state.new_image:
                buf = BytesIO()
                st.session_state.new_image.save(buf, format="png")
                download_button = st.download_button(label="Download", data=buf.getvalue(), file_name="fitting_result.png", mime="image/png", use_container_width=True)
            else:
                download_button = st.button("Download", key="download", use_container_width=True)