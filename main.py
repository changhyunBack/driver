import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from pyparsing import empty
import matplotlib.pyplot as plt # 새로추가 
import os 
import matplotlib.font_manager as fm

def fontRegistered():
    font_dirs = [os.getcwd() + '/customFonts']
    font_files = fm.findSystemFonts(fontpaths=font_dirs)

    for font_file in font_files:
        fm.fontManager.addfont(font_file)
    fm._load_fontmanager(try_read_cache=False)
fontRegistered()
fontNames = [f.name for f in fm.fontManager.ttflist]
#fontname = st.selectbox("폰트 선택", unique(fontNames))
fontname=fontNames[0]
plt.rc('font', family=fontname)

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
        sizes=prediction.ravel()
        wedgeprops = {'width':0.6, 'linewidth':1, 'edgecolor':'black'}
        
        fig1, ax1 = plt.subplots(figsize=(5,5))
        ax1.pie(sizes,labels=labels,autopct='%1.1f%%',wedgeprops=wedgeprops,pctdistance=0.7,textprops={'fontsize': 12},labeldistance=1.1)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax1.set_facecolor('none')
        st.pyplot(fig1)
        
        
    with con3:
            # st.write("Uploaded DataFrame:")
            # #st.write(df.head(12))
            # st.write(df.head(50).style.set_table_styles([{'selector': 'tr', 'props': [('max-height', '400px'),('max-column','400px')]}]))
            num_rows_to_display = len(df)
            st.write(f"Uploaded DataFrame : {num_rows_to_display}")
            st.write(df.head(num_rows_to_display).style.set_table_styles([{'selector': 'tr', 'props': [('max-height', '400px'), ('max-column', '400px')]}]))
            st.markdown(f"<h3 style='margin-left: 150px; margin-right: 0px;'> 운전자는 {round(pred_prob * 100,2)}%로 [{pred}] 입니다</h3>", unsafe_allow_html=True)

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
# labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
# sizes = [15, 30, 45, 10]
# explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

# fig1, ax1 = plt.subplots()
# ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
#         shadow=True, startangle=90)
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# st.pyplot(fig1)
