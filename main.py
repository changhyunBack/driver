import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from pyparsing import empty

st.set_page_config(layout="wide")
# con1 : 제목, 파일 넣기 
empty1,con1,empty2 = st.columns([0.3,0.4,0.3])
# con2 : 그래프 넣기 
empty1,con2,conf5,empty2 = st.columns([0.1,0.1,0.75,0.1])
# con3 : 데이터프레임,  con4 : 모델결과
empty1,con3,con4,empty2 = st.columns([0.1,0.5,0.5,0.1])

# 1분 ~ 3분짜리 모델 불러오기.
model1=tf.keras.models.load_model("a5.h5")
scaler1=joblib.load("a5.pkl")

model2=tf.keras.models.load_model("d7.h5")
scaler2=joblib.load("d7.pkl")

model3=tf.keras.models.load_model("s8.h5")
scaler3=joblib.load("s8.pkl")

# 윗줄 공백 
with empty1:
    empty()
# 제목과 파일 넣기
with con1:
    st.write("""
    # Driver Classification
    """)
    file = st.file_uploader("Pick a file")
    
if file is not None:
    df =pd.read_csv(file)
    df=df.loc[:,['Timestamp','AcX', 'AcY', 'AcZ', 'GyX', 'GyY','GyZ', 'Speed', 'Heading', 'CAN_speed', 'CAN_accelPosition', 'CAN_RPM']]
    # 그래프 삽입 
    with con2:
        genre = st.radio(
            "select the column",
            ['AcX', 'AcY', 'AcZ', 'GyX', 'GyY','GyZ', 'Speed', 'Heading', 'CAN_speed', 'CAN_accelPosition', 'CAN_RPM'],index=None
        )
   
    df['time'] = pd.to_datetime(df['Timestamp'])
    df['time'] = df['time'].dt.strftime('%H:%M:%S')
    with conf5:
        if genre:
            st.line_chart(df,x='time', y=genre)
    
    with con3:
        df.drop(columns=['Timestamp','time'],inplace=True)
        x=df.values
        if len(x)==60:
            x=scaler1.transform(x).reshape(1,60,11)
            prediction=model1.predict(x)
        elif len(x)==120:
            x=scaler2.transform(x).reshape(1,120,11)
            prediction=model2.predict(x)
        elif len(x)==180:
            x=scaler3.transform(x).reshape(1,180,11)
            prediction=model3.predict(x) 
        else:
            st.write('## Please Enter the collect data')

        result = pd.DataFrame(prediction)
        result= result.melt()
        result['variable']=result['variable'].map({0:'창현',1:'도윤',2:'현선'})
        text="probability value"
        st.markdown(f"<h3 style='margin-left: 285px; margin-right: 20px;'>{text}</h3>", unsafe_allow_html=True)
        st.bar_chart(result,x='variable',y='value')
        
        pred= np.argmax(prediction,1)
        pred_prob=float(prediction.ravel()[pred])
        
        if pred ==0:
            pred='창현'
        elif pred==1:
            pred='도윤'
        elif pred==2:
            pred='현선'
        else:
            pred='Known'

        st.markdown(f"<h3 style='margin-left: 200px; margin-right: 20px;'> 운전자는 {round(pred_prob * 100,2)}%로 '{pred}' 입니다</h3>", unsafe_allow_html=True)
        #st.write(f'### 운전자는 "{pred}" 입니다')
        with con4:
                st.write("Uploaded DataFrame:")
                st.write(df.head(7))

    