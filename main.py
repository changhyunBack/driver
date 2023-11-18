import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from pyparsing import empty
import matplotlib.pyplot as plt # 새로추가 
import plotly.express as px

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
    
    # chart 
    with conf5:
        if genre:
            st.line_chart(df,x='time', y=genre)
    
    # model predict
    with con4:
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


        # 단일 예측값
        pred= np.argmax(prediction,1)
        # 예측 확률값 
        pred_prob=float(prediction.ravel()[pred])
        if pred ==0:
            pred='창현'
        elif pred==1:
            pred='도윤'
        elif pred==2:
            pred='현선'
        else:
            pred='Known'
        
        labels=['창현','도윤','현선']
        prob=prediction.ravel()
        chart_data={'labels':labels, 'prob':prob}
        #chart_data=px.data.tips()
        fig=px.pie(chart_data,values='prob',names='labels')
        fig.update_traces(hole=.4,textfont_size=20)
        # 라벨 크기 및 그래프 크기 조정
        fig.update_layout(
            title=dict(text='Probability', font=dict(size=20)),  # 타이틀 폰트 크기 조정
            showlegend=True,  # 범례 표시 여부
            legend=dict(title=dict(text='Categories', font=dict(size=15))),  # 범례 폰트 크기 조정
            font=dict(size=12),  # 라벨 폰트 크기 조정
            margin=dict(l=0, r=0, b=0, t=30),  # 그래프의 여백 조정
        )

        st.plotly_chart(fig)
        
        
    with con3:
            # st.write("Uploaded DataFrame:")
            # #st.write(df.head(12))
            # st.write(df.head(50).style.set_table_styles([{'selector': 'tr', 'props': [('max-height', '400px'),('max-column','400px')]}]))
            num_rows_to_display = len(df)
            st.write(f"Uploaded DataFrame : {num_rows_to_display}")
            st.write(df.head(num_rows_to_display).style.set_table_styles([{'selector': 'tr', 'props': [('max-height', '200px'), ('max-column', '200px')]}]))
            st.markdown(f"<h3 style='margin-left: 150px; margin-right: 0px;'> 운전자는 {round(pred_prob * 100,2)}%로 [{pred}] 입니다</h3>", unsafe_allow_html=True)

