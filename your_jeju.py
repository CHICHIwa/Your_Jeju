import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
from folium import IFrame
from streamlit import components
from plotly.graph_objs import Figure
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import pickle
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import linear_kernel
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
import pickle
import os
from haversine import haversine



@st.cache_data()
def load_data(path, encoding='utf-8'):  # Default encoding set to 'utf-8'
    data = pd.read_csv(path, encoding=encoding)  # Apply encoding
    return data


# 타이틀, 아이콘, 레이아웃 설정
st.set_page_config(
    page_title="제주도 관광의 A to Z - 데이터 기반 관광 분석",
    page_icon="🍊",
    layout="wide"
)

# HTML/CSS를 사용하여 특정 텍스트에 Jaljali 폰트 적용
st.markdown("""
    <style>
        /* JalnanGothic 폰트를 여기에 지정 */
        .JalnanGothicTTF-text {
            font-family: 'JalnanGothicTTF', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

# CSS를 사용하여 전체 페이지의 배경 이미지 설정
def set_bg_image():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdN82HY%2FbtsGK99f1pm%2FdW4DfXw42gpvIxOQon7RRK%2Fimg.webp");
            background-size: cover;
            background-position: center center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_image()  # 배경 이미지 함수 호출


# 대시보드에 페이지 제목 설정
def add_page_title():
    st.title("🍊제주도 관광 데이터 분석 Final-Project🍊")


# Page 클래스 정의 / 각 페이지 나타내고 제목, 내용,데이터프레임, 그래프, 이미지를 속성으로 가짐
class Page:
    def __init__(self, title, content, dfs=None, graphs=None, images=None, df_titles=None, graph_descriptions=None, image_title=None, functions=None):
        self.title = title
        self.content = content
        self.dfs = dfs if dfs is not None else []
        self.graphs = graphs if graphs is not None else []
        self.images = images if images is not None else []
        self.df_titles = df_titles if df_titles is not None else []
        self.graph_descriptions = graph_descriptions if graph_descriptions is not None else []
        self.image_title = image_title if image_title is not None else []
        self.functions = functions if functions is not None else []  # 함수를 위한 필드 추가

Accommodation_Facility_Information_df1 = pd.read_csv("Jeju/군집분석/호텔(스케일링전).csv", encoding='cp949')
Accommodation_Facility_Information_df2 = pd.read_csv("Jeju/군집분석/호텔(스케일링후).csv", encoding='cp949')


# 페이지 함수 생성
def accommodation_analysis_page():
    st.subheader("스케일링 전")
    
    # 첫 번째 데이터프레임 그래프
    plt.figure(figsize=(20, 20))
    sns.pairplot(Accommodation_Facility_Information_df1[['1Q Average Price', '2Q Average Price', 'Number of rooms', 'Number of Accommodations Nearby']])
    st.pyplot(plt)

    st.subheader("(log, Robust, 표준화 스케일링 후")
        
    # 두 번째 데이터프레임 그래프
    plt.figure(figsize=(20, 20))
    sns.pairplot(Accommodation_Facility_Information_df2[['1Q Average Price_standard', '2Q Average Price_standard', 'Number of rooms_standard', 'Number of Accommodations Nearby_standard']])
    st.pyplot(plt)



###############################################################################
@st.cache_data()
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

indices = load_model("Jeju/indices.pkl")
cosine_sim = load_model("Jeju/cosine_sim.pkl")
final_city_review = load_model("Jeju/final_city_review.pkl")
tfidf_matrix = load_model("Jeju/tfidf_matrix.pkl")
tfidf = load_model("Jeju/tfidf.pkl")



# 추천1) 제주시 
def get_user_input_vector_city(user_input, tfidf_model):
    return tfidf_model.transform([user_input])


def get_recommendations_by_user_input_with_hotel_city(user_input, hotel_name, tfidf_model, cosine_sim=cosine_sim):
    # 호텔에 부합하는 행들 필터링
    hotel_indices_city = final_city_review[final_city_review['숙박업명'] == hotel_name].index

    # TF-IDF 벡터 생성
    user_tfidf_vector_city = get_user_input_vector_city(user_input, tfidf_model)

    # 사용자 입력과 호텔 필터링을 고려한 코사인 유사도 계산
    cosine_sim_user_city = linear_kernel(user_tfidf_vector_city, tfidf_matrix[hotel_indices_city])

    # 유사도가 높은 순으로 정렬
    sim_scores_city = list(enumerate(cosine_sim_user_city[0]))
    sim_scores_city = sorted(sim_scores_city, key=lambda x: x[1], reverse=True)

    # 상위 5개 식당 추출
    sim_scores_city = sim_scores_city[:5]
    restaurant_indices_city = [hotel_indices_city[i[0]] for i in sim_scores_city]

    # 추천 식당과 유사도 반환
    recommended_restaurants_city = final_city_review.iloc[restaurant_indices_city][['식당명', '검색량합계값', '숙박_식당 거리']]
    similarity_scores = [round(i[1], 3) for i in sim_scores_city]

    return recommended_restaurants_city, similarity_scores


# 사용자에게 식당 추천하는 함수


def recommend_restaurant_city():
    st.subheader('> 제주시')

    # 중복 제거한 숙박업명 목록 생성
    unique_hotels = set(final_city_review['숙박업명'].values)

    # 사용자가 선택할 수 있는 드롭다운 메뉴 생성
    user_hotel = st.selectbox("어느 호텔에서 묵고 계신가요?", sorted(unique_hotels))

    # 사용자가 호텔을 선택하지 않았을 경우
    if not user_hotel:
        st.warning("호텔을 선택해주세요.")
        return

    # 사용자 입력 받기
    user_input = st.text_input("어떤 식당을 찾으시나요? ")

    # 호텔과 사용자 입력에 기반한 식당 추천 및 유사도 가져오기
    recommended_restaurants, similarity_scores = get_recommendations_by_user_input_with_hotel_city(user_input, user_hotel, tfidf, cosine_sim)

    if recommended_restaurants.empty:
        #print("입력하신 조건에 부합하는 식당이 없습니다.")
        st.write("입력하신 조건에 부합하는 식당이 없습니다.")
    elif user_hotel and user_input:
        with st.container():
            st.info("입력하신 조건과 호텔에 부합하는 식당을 아래와 같이 추천드립니다:")
            for idx, (restaurant, search_count, distance) in enumerate(recommended_restaurants.values):
                distance = round(distance, 2)
                score = similarity_scores[idx]
                st.write(f"### {restaurant}")
                st.write(f"**유사도:** {score}")
                st.write(f"**식당 검색량:** {search_count} 건")
                st.write(f"**숙박-식당 거리:** {distance} km")
                st.write("---")  # 각 식당의 정보를 구분하기 위해 수평 선 추가





indices_1 = load_model("Jeju/indices_1.pkl")
cosine_sim_1 = load_model("Jeju/cosine_sim_1.pkl")
final_downtown_review_1 = load_model("Jeju/final_downtown_review.pkl")
tfidf_matrix_1 = load_model("Jeju/tfidf_matrix_1.pkl")
tfidf_1 = load_model("Jeju/tfidf_1.pkl")
    
# 추천2) 서귀포시
def get_user_input_vector(user_input, tfidf_model):
    return tfidf_model.transform([user_input])

def get_recommendations_by_user_input_with_hotel_downtown(user_input, hotel_name, tfidf_model, cosine_sim=cosine_sim_1):
    # 호텔에 부합하는 행들 필터링
    hotel_indices = final_downtown_review[final_downtown_review['숙박업명'] == hotel_name].index

    # Tfidf 백터생성
    user_tfidf_vector = get_user_input_vector(user_input, tfidf_model)

    # 사용자입력 & 호텔 필터링 코사인 유사도 계산
    cosine_sim_user = linear_kernel(user_tfidf_vector, tfidf_matrix_1[hotel_indices])

    # 정렬 (유사도 높은순)
    sim_scores = list(enumerate(cosine_sim_user[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 상위 5개 식당 추출
    sim_scores = sim_scores[:5]
    restaurant_indices = [hotel_indices[i[0]] for i in sim_scores]

    # 추천 식당과 유사도 반환
    recommended_restaurants = final_downtown_review.iloc[restaurant_indices][['식당명', '검색량합계값', '숙박_식당 거리']]
    similarity_scores = [round(i[1], 3) for i in sim_scores]

    return recommended_restaurants, similarity_scores


# 사용자에게 식당 추천하는 함수
def recommend_restaurant_downtown():
    st.subheader('> 서귀포시')
    # 중복 제거한 숙박업명 목록 생성
    unique_hotels = set(final_downtown_review['숙박업명'].values)

    # 사용자가 선택할 수 있는 드롭다운 메뉴 생성
    user_hotel = st.selectbox("어느 호텔에서 묵고 계신가요?", sorted(unique_hotels))

    # 사용자가 호텔을 선택하지 않았을 경우
    if not user_hotel:
        st.warning("호텔을 선택해주세요.")
        return

    #user_input = input("어떤 식당을 찾으시나요? ")
    user_input = st.text_input("어떤 식당을 찾으시나요? ")

    # 호텔과 사용자 입력에 기반한 식당 추천 및 유사도 가져오기
    recommended_restaurants, similarity_scores = get_recommendations_by_user_input_with_hotel_downtown(user_input, user_hotel, tfidf_1, cosine_sim_1)

    if recommended_restaurants.empty:
        #print("입력하신 조건에 부합하는 식당이 없습니다.")
        st.write("입력하신 조건에 부합하는 식당이 없습니다.")
    elif user_hotel and user_input:
        with st.container():
            st.info("입력하신 조건과 호텔에 부합하는 식당을 아래와 같이 추천드립니다:")
            for idx, (restaurant, search_count, distance) in enumerate(recommended_restaurants.values):
                distance = round(distance, 2)
                score = similarity_scores[idx]
                st.write(f"### {restaurant}")
                st.write(f"**유사도:** {score}")
                st.write(f"**식당 검색량:** {search_count} 건")
                st.write(f"**숙박-식당 거리:** {distance} km")
                st.write("---")  # 각 식당의 정보를 구분하기 위해 수평 선 추가




#############################################################################

def add_future_plans_page():
    st.write("""
    ## 마무리
    ### 그동안 고생하신 매니저님들과 튜터님들, 수강생분들 모두 고생하셨습니다!
    """)
    
    # 이미지 파일을 로드합니다 (로컬 경로를 사용)
    image = Image.open("Jeju/bye.png")
    
    # 이미지를 스트림릿 페이지에 표시
    st.image(image, caption='한라산의 울림, 바다의 속상임 - 제주도에서 휴식을 즐겨보세요')

















def show_cluster_names():
    # 군집 번호 선택을 위한 드롭다운 메뉴 생성
    cluster_number = st.selectbox('Select Cluster Number:', sorted(merged_df['kmeans_cluster'].unique()))

    # 선택된 군집 번호에 해당하는 숙박업명들을 필터링
    cluster_data = merged_df[merged_df['kmeans_cluster'] == cluster_number]['숙박업명']
    
    # 결과를 화면에 표시
    st.write('Accommodation Facility Names in Cluster {}:'.format(cluster_number))
    st.write(cluster_data)

def show_cluster_descriptions():
    # 클러스터 설명
    cluster_descriptions = {
        0: ("모든 숙박시설에 주차장이 있고, 대부분 바와 카페를 보유. "
            "야외수영장, 스파, 사우나가 일반적. "
            "이 클러스터는 리조트 스타일의 숙박시설을 대표할 가능성이 높음."),
        1: ("주차장 존재율이 높으나 100%는 아님. "
            "식당, 바, 카페 보유율이 상대적으로 낮고, 야외수영장과 스파는 거의 없음. "
            "이 클러스터는 기본적인 편의 시설을 제공하는 저가 또는 중가 호텔로 추정됨."),
        2: ("모든 숙박시설에 주차장이 있고, 식당과 바의 존재율이 비교적 높음. "
            "비즈니스 센터와 연회장 존재율도 높음, 대형 호텔 또는 비즈니스 호텔일 가능성이 있음."),
        3: ("모든 숙박시설에 주차장이 있으며, 식당, 바, 카페 보유율이 높음. "
            "스파와 사우나는 드물게 존재. "
            "주로 표준적인 시설을 갖춘 숙박시설로 구성."),
        4: ("모든 숙박시설에 주차장이 있고, 모든 숙박시설에 식당, 바, 카페, 스파, 사우나, 야외수영장 존재. "
            "럭셔리 호텔이나 고급 리조트를 대표."),
        5: ("모든 숙박시설에 주차장, 식당, 바, 카페 존재. "
            "야외수영장과 스파는 드물게 존재. "
            "대형 또는 고급 시설을 갖춘 숙박시설이 포함될 가능성이 있음, 일부 럭셔리 요소 포함.")
        
    }
    
    # 군집 번호 선택을 위한 드롭다운 메뉴 생성
    cluster_number = st.selectbox('Select Cluster Number:', sorted(merged_df['kmeans_cluster'].unique()))
    
    # 선택된 군집 번호에 해당하는 숙박업명들을 필터링
    cluster_data = merged_df[merged_df['kmeans_cluster'] == cluster_number]['숙박업명']
    st.write('Accommodation Facility Names in Cluster {}:'.format(cluster_number))
    st.dataframe(cluster_data, hide_index=True)
    # 선택된 클러스터에 대한 설명을 보여주는 expander
    with st.expander("클러스터 설명 보기"):
        st.write(cluster_descriptions.get(cluster_number, "선택한 클러스터에 대한 설명이 없습니다."))

    








def accommodation_umap_kmeans_page():
    st.plotly_chart(fig26)
        
    show_cluster_descriptions()








def show_pages(pages):
    for page in pages:
        if isinstance(page, Page):
            st.write(f"## {page.title}")
            st.write(page.content)
            for func in page.functions:
                func()
            for i, df in enumerate(page.dfs):
                if df is not None:
                    st.write(f"> **{page.df_titles[i]}**" if i < len(page.df_titles) else "> **Data**")
                    st.dataframe(df, use_container_width=True)
            for image in page.images:
                if image:
                    st.image(image, use_column_width=True)            
            # "관광 현황 분석" 페이지에 대한 특별한 레이아웃 처리
            if page.title in ["관광 현황 - 동반자 유형별 분석", "농협카드 - 시계열 모델링"]:
                # 첫 번째 그래프는 전체 너비로 표시
                if page.graphs:
                    if isinstance(page.graphs[0], Figure):
                        st.plotly_chart(page.graphs[0], use_container_width=True)
                        if len(page.graph_descriptions) > 0:
                            st.write(page.graph_descriptions[0])  # 첫 번째 그래프의 설명 추가
                    else:
                        st.error("Invalid graph object detected.")

                # 그 이후 그래프를 두 개씩 나열
                col_index = 0
                cols = [None, None]  # 두 개의 열을 위한 임시 리스트
                for i, graph in enumerate(page.graphs[1:]):  # 첫 번째 그래프를 제외하고 시작
                    if col_index == 0:
                        cols = st.columns(2)  # 두 열 생성
                    if isinstance(graph, Figure):
                        cols[col_index].plotly_chart(graph, use_container_width=True)
                        if i + 1 < len(page.graph_descriptions):  # 설명이 있으면 출력
                            cols[col_index].write(page.graph_descriptions[i + 1])
                    else:
                        cols[col_index].error("Invalid graph object detected.")
                    
                    col_index = (col_index + 1) % 2  # 0, 1, 0, 1, ...으로 변경하여 열을 번갈아 선택
            elif page.title == '제주도의 미래 소비 예측을 위한 Prophet모델링':
                for i, graph in enumerate(page.graphs):
                    if isinstance(graph, Figure):
                        st.plotly_chart(graph, use_container_width=True)
                        if i < len(page.graph_descriptions):
                            st.write(page.graph_descriptions[i])
               # 마지막으로 이미지를 표시
                if page.images:
                    st.image(present_jeju, use_column_width=True)
            
            elif page.title == '숙박 리뷰 키워드_호텔 점수 산정':
                st.write("> 2023 제주 숙박 키워드")
                st.image(wordcloud_pos_review, use_column_width=True)
                st.write("**작년 한 해동안 제주도 숙박 시설 리뷰에서 많이 언급된 긍정 리뷰 목록**")
                
                # 그래프와 설명 추가
                graphs_and_descriptions = [
                    (fig27, "각 키워드의 출현 빈도를 전체 키워드의 출현 총계로 나누어서, 각 키워드에 대한 점수에 빈도 비율에 해당하는 가중치를 부여한 점수 분포"),
                    (fig31, "리뷰를 가진 제주도 호텔 40곳을 뽑아, 가중치 점수를 반영하여 각 호텔별 키워드 점수를 산출한 통계"),
                    (fig32, "그 중 제주시/서귀포시 두 구역으로 나누어 점수가 높은 5곳의 호텔을 각각 선정")
                ]
                
                for i, (fig, description) in enumerate(graphs_and_descriptions):
                    st.plotly_chart(fig)
                    st.write(f"**{description}**")
                        
            elif page.title == "지역별 상위 5개 호텔 & 식당 분포":
                for graph_info in page.graphs:
                    if isinstance(graph_info, tuple) and len(graph_info) == 2:
                        graph, description = graph_info
                        if isinstance(graph, folium.Map):
                            if description:
                                st.write(f"> **{description}**") 
                            folium_static(graph, width=1000, height=500)
                        else:
                            st.error("Invalid graph object detected for display.")
                    elif isinstance(graph_info, (folium.Map, go.Figure)):
                        if isinstance(graph_info, folium.Map):
                            folium_static(graph_info, width=1400, height=500)
                        elif isinstance(graph_info, go.Figure):
                            st.plotly_chart(graph_info, use_container_width=True, width=800)
                    else:
                        st.error("Invalid graph object detected for display.")
                        
                                        
            elif page.title == "식당 추천시스템_제주시":
                recommend_restaurant_city()
                
            elif page.title == '식당 추천시스템_서귀포시':
                recommend_restaurant_downtown()
                                                           
            elif page.title == "분류별 추천 관광지":
                for graph in page.graphs:
                    if isinstance(graph, folium.Map):
                        folium_static(graph, width=1000, height=800)
                    else:
                        st.error("Invalid graph object detected for the map display.")
            elif page.title == "숙박시설 - 제주호텔 군집분석":
                accommodation_analysis_page()
            elif page.title == "숙박시설 - UMAP & K-Means":
                accommodation_umap_kmeans_page()
            elif page.title == "끝이 다가오는 것은 시작을 알리는 신호입니다":
                add_future_plans_page()                                  
            else:
                # 다른 페이지들은 모든 그래프를 두 개씩 나열
                col_index = 0
                cols = [None, None]
                for i, graph in enumerate(page.graphs):
                    if col_index == 0:
                        cols = st.columns(2)
                    if isinstance(graph, Figure):
                        cols[col_index].plotly_chart(graph, use_container_width=True)
                        if i < len(page.graph_descriptions):  # 설명이 있으면 출력
                            cols[col_index].write(page.graph_descriptions[i])
                    else:
                        cols[col_index].error("Invalid graph object detected.")
                    
                    col_index = (col_index + 1) % 2

        elif isinstance(page, Section):
            st.write(f"## {page.title}")
        else:
            st.warning("Unknown page type!")
            

class Section:
    def __init__(self, title):
        self.title = title


# 원본 데이터 로딩
df_1 = load_data("Jeju/데이터/제주 동반자 유형별 여행 계획 데이터.csv")
df_2 = load_data("Jeju/데이터/제주 무장애 관광지 입장 데이터.csv") 
df_3 = load_data("Jeju/데이터/SNS 제주 관광 키워드별 수집 통계_월.csv") 
df_4 = load_data("Jeju/데이터/제주 관광수요예측 데이터_비짓제주 로그 데이터.csv")
df_5 = load_data("Jeju/데이터/제주관광공사 관광 소비행테 데이터 카드사 음식 급상승 데이터.csv", encoding='cp949')
df_6 = load_data("Jeju/데이터/[NH농협카드] 일자별 소비현황_제주.csv")
df_7 = load_data("Jeju/데이터/종합맵.csv")
####################################################################
Consumption_status_by_date_NH = pd.read_csv("Jeju/데이터/[NH농협카드] 일자별 소비현황_제주.csv", parse_dates=['승인일자'], index_col='승인일자')
####################################################################
#계절성 분석
# Assuming 'Consumption_status_by_date_NH' is pre-loaded with your data
consumption_data = Consumption_status_by_date_NH['이용금액_전체']

# Perform seasonal decomposition
result = seasonal_decompose(consumption_data, model='additive', period=365)

# Convert the seasonal component to a DataFrame and reset index to 'date'
seasonal_df = pd.DataFrame(result.seasonal).reset_index()
seasonal_df.columns = ['date', 'seasonal']  # Rename columns appropriately

# Visualize the seasonal component using Plotly Express
fig1 = px.line(seasonal_df, x='date', y='seasonal', title='Seasonal Component of Consumption',
              labels={'seasonal': 'Seasonality'}, template='plotly_dark')
############################################

#추세 분석
Consumption_status_by_date_NH['7_day_rolling_avg'] = Consumption_status_by_date_NH['이용금액_전체'].rolling(window=7).mean()
Consumption_status_by_date_NH['30_day_rolling_avg'] = Consumption_status_by_date_NH['이용금액_전체'].rolling(window=30).mean()

# Create a figure using Plotly graph objects
fig2 = go.Figure()

# Add traces for the original data and the rolling averages
fig2.add_trace(go.Scatter(x=Consumption_status_by_date_NH.index, y=Consumption_status_by_date_NH['이용금액_전체'], mode='lines', name='Original'))
fig2.add_trace(go.Scatter(x=Consumption_status_by_date_NH.index, y=Consumption_status_by_date_NH['7_day_rolling_avg'], mode='lines', name='7 Day Rolling Average'))
fig2.add_trace(go.Scatter(x=Consumption_status_by_date_NH.index, y=Consumption_status_by_date_NH['30_day_rolling_avg'], mode='lines', name='30 Day Rolling Average'))

# Update the layout of the figure
fig2.update_layout(
    title='Daily 이용금액_전체 with Rolling Average',
    xaxis_title='Date',
    yaxis_title='Consumption',
    template='plotly_dark'
)
#################################################
nlags = int(len(Consumption_status_by_date_NH) * 0.1) 
#정상성 분석
acf_values = acf(Consumption_status_by_date_NH['이용금액_전체'], fft=False, nlags=nlags)  # Ensure the column name is correct

# Create a list of lag values
lags = list(range(len(acf_values)))

# Create a Plotly figure
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=lags, y=acf_values, mode='lines+markers', name='ACF'))

# Update the layout of the figure
fig3.update_layout(
    title='Autocorrelation Function',
    xaxis_title='Lags',
    yaxis_title='ACF',
    template='plotly_dark'
)
######################################################

#노이즈 분석
rolling_window = 7  # For example, using 12 points for moving average was mentioned
Consumption_status_by_date_NH['smoothed'] = Consumption_status_by_date_NH['이용금액_전체'].rolling(window=rolling_window).mean()

# Create a Plotly figure
fig4 = go.Figure()

# Add trace for original data
fig4.add_trace(go.Scatter(
    x=Consumption_status_by_date_NH.index,  # Or you might use a 'Date' column if available
    y=Consumption_status_by_date_NH['이용금액_전체'],
    mode='lines',
    name='Original Data'
))

# Add trace for smoothed data
fig4.add_trace(go.Scatter(
    x=Consumption_status_by_date_NH.index,  # Or 'Date' column
    y=Consumption_status_by_date_NH['smoothed'],
    mode='lines',
    name='Smoothed Data',
    line=dict(color='red')
))

# Update the layout of the figure
fig4.update_layout(
    title='Time Series with Smoothing',
    xaxis_title='Time',
    yaxis_title='Value',
    template='plotly_dark'
)
###########################
@st.cache_data()
def load_model1(model_path):
    with open(model_path, 'rb') as f:
        model1 = pickle.load(f)
    return model1

def make_forecast1(model1):
    future1 = model1.make_future_dataframe(periods=180)
    forecast1 = model1.predict(future1)
    return forecast1

model1 = load_model1("Jeju/prophet_model(초안).pkl")
forecast1 = make_forecast1(model1)

# 예측 그래프 표시
fig5 = plot_plotly(model1, forecast1)
fig5.update_layout(title='초기 Prophet 모델을 활용한 6개월간 소비시장 예측')
# 컴포넌트별 시각화
components_fig5 = plot_components_plotly(model1, forecast1)
###################################################################
@st.cache_data()
def load_model2(model_path):
    with open(model_path, 'rb') as f:
        model2 = pickle.load(f)
    return model2

def make_forecast2(model2):
    future2 = model2.make_future_dataframe(periods=180)
    forecast2 = model2.predict(future2)
    return forecast2

model2 = load_model2("Jeju/prophet_model(최종).pkl")
forecast2 = make_forecast2(model2)

# 예측 그래프 표시
fig6 = plot_plotly(model2, forecast2)
fig6.update_layout(title='최종 Prophet 모델을 활용한 6개월간 소비시장 예측')
# 컴포넌트별 시각화
components_fig6 = plot_components_plotly(model2, forecast2)
#######################################################
present_jeju = Image.open("Jeju/제주도 현황.png")
#########################################################
cl_nm_counts = load_data("Jeju/시각화/cl_nm_counts.csv")
df_top_keywords = load_data("Jeju/시각화/df_top_keywords.csv")
df_top_CNTNTSs = load_data("Jeju/시각화/df_top_CNTNTSs.csv")
Sum_df = load_data("Jeju/시각화/Sum_df.csv")
sorted_group_df = load_data("Jeju/시각화/sorted_group_df.csv")
###################################################################
def format_period(period):
    year, month = divmod(period, 100)
    return f"{year}년 {month}월"

fig7 = go.Figure()

# Add a trace for each investigation period
for 조사기간 in cl_nm_counts['조사기간'].unique():
    filtered_df = cl_nm_counts[cl_nm_counts['조사기간'] == 조사기간]
    fig7.add_trace(
        go.Bar(
            visible=False,
            name=f"조사기간: {format_period(조사기간)}",
            x=filtered_df['동반자유형'],
            y=filtered_df['비율(%)']
        )
    )

# Make the first trace visible
fig7.data[0].visible = True

# Create sliders
steps = []
for i, 조사기간 in enumerate(cl_nm_counts['조사기간'].unique()):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig7.data)},
              {"title": f"조사기간: {format_period(조사기간)}"}],  # layout attribute
        label=format_period(조사기간)  # slider label
    )
    step["args"][0]["visible"][i] = True  # Toggle visibility of the i'th trace
    steps.append(step)

sliders = [dict(
    active=0,
    currentvalue={"prefix": "조사기간 선택: "},
    pad={"t": 50},
    steps=steps
)]

# Annotations for data source
annotations = [dict(
    showarrow=False,
    xref="paper",
    yref="paper",
    x=0,
    y=-0.1,
    xanchor="left",
    yanchor="top",
    font=dict(size=12)
)]

# Update layout for Y-axis title and sliders
fig7.update_layout(
    yaxis_title='비율(%)',
    sliders=sliders,
    title="조사기간별 동반자유형 분석",
    annotations=annotations
)
#######################################
fig8 = go.Figure()

# Add a pie chart for each companion type
for i, cl_nm in enumerate(df_top_keywords['동반자유형'].unique()):
    df_filtered = df_top_keywords[df_top_keywords['동반자유형'] == cl_nm]
    keywords = df_filtered['키워드'].tolist()
    frequencies = df_filtered['빈도'].tolist()

    fig8.add_trace(
        go.Pie(
            labels=keywords,
            values=frequencies,
            name=cl_nm,
            visible=(i == 0)  # Only the first companion type is visible initially
        )
    )

# Create slider steps
steps = []
for i, cl_nm in enumerate(df_top_keywords['동반자유형'].unique()):
    step = dict(
        method='update',
        args=[{'visible': [(j == i) for j in range(len(df_top_keywords['동반자유형'].unique()))]},
              {'title': f'동반자 유형: {cl_nm}'}],
        label=cl_nm
    )
    steps.append(step)

# Set up the sliders and annotations in the layout
fig8.update_layout(
    sliders=[dict(
        active=0,
        currentvalue={'prefix': '동반자 유형: '},
        steps=steps
    )],
    annotations=[dict(
        showarrow=False,
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.2,
        xanchor="center",
        yanchor="top",
        font=dict(size=12)
    )],
    title="동반자 유형별 상위 키워드 분석"
)
####################################################
fig10 = px.line(Sum_df, x='방문기간', y='입장인원수', color='관광지명', 
                title='방문기간별 관광지 입장인원수',
                labels={'방문기간': '방문 기간', '입장인원수': '입장 인원수', '관광지명': '관광지 명'})

# Update graph layout
fig10.update_layout(
    xaxis_title='방문 기간',
    yaxis_title='입장 인원수',
    legend_title='관광지'
)
###################################################3
fig11 = go.Figure()

unique_entry_types = sorted_group_df['입장구분명'].unique()

# Add a bar for each entry type
for entry_type in unique_entry_types:
    filtered_df = sorted_group_df[sorted_group_df['입장구분명'] == entry_type]
    fig11.add_trace(
        go.Bar(
            x=filtered_df['관광지명'],
            y=filtered_df['입장인원수'],
            name=entry_type,
            visible=False  # Start with all bars hidden, will enable visibility below
        )
    )

# Setup buttons for the interactive component
buttons = []
for i, entry_type in enumerate(unique_entry_types):
    visibility = [False] * len(unique_entry_types)
    visibility[i] = True
    buttons.append(
        dict(
            label=entry_type,
            method="update",
            args=[{"visible": visibility},
                  {"title": f"{entry_type} - 관광지별 입장인원수"}]
        )
    )

# Configure layout with buttons
fig11.update_layout(
    updatemenus=[{
        "buttons": buttons,
        "direction": "down",
        "pad": {"r": 10, "t": 10},
        "showactive": True,
        "x": 0.1,
        "xanchor": "left",
        "y": 1.2,
        "yanchor": "top"
    }],
    title=f"{unique_entry_types[0]} - 관광지별 입장인원수"
)

# Initially set the first dataset to visible
fig11.data[0].visible = True
#####################################################
year_df2 = load_data("Jeju/시각화/year_df2.csv")
sns_df2 = load_data("Jeju/시각화/sns_df2.csv")
top_seasons = load_data("Jeju/시각화/top_seasons.csv")
top10_classification_df = load_data("Jeju/시각화/top10_classification_df.csv")
#####################################################################
year_df2['게시년월'] = year_df2['게시년월'].astype(str)

fig13 = go.Figure()

# Create unique months from the DataFrame
unique_months = year_df2['게시년월'].unique()

# Add traces for each month and keyword, initially hidden
for month in unique_months:
    for spot in year_df2[year_df2['게시년월'] == month]['대표키워드명'].unique():
        filtered_df = year_df2[(year_df2['게시년월'] == month) & (year_df2['대표키워드명'] == spot)]
        fig13.add_trace(
            go.Bar(
                x=[spot],
                y=filtered_df['대표키워드언급수'],
                name=spot,
                visible=False,  # initially all traces are hidden
                legendgroup=month,  # group by month for toggling
                legendgrouptitle_text=month  # show month as group title
            )
        )

# Create buttons for each month to toggle visibility
buttons = []

for i, month in enumerate(unique_months):
    visibility = [month == trace.legendgroup for trace in fig13.data]  # adjust visibility based on group

    buttons.append(
        dict(
            label=month,
            method="update",
            args=[{"visible": visibility},
                  {"title": f"{month} - 키워드별 언급수"}]
        )
    )

# Add dropdown menu with buttons to the layout
fig13.update_layout(
    updatemenus=[dict(
        active=0,
        buttons=buttons,
        direction="down",
        pad={"r": 10, "t": 10},
        showactive=True,
        x=0.1,
        xanchor="left",
        y=1.2,
        yanchor="top"
    )],
    title=f"{unique_months[0]} - 키워드별 언급수"
)

# Set the visibility of the first month's data as default
for trace in fig13.data:
    trace.visible = trace.legendgroup == unique_months[0]
##########################################################
fig15 = go.Figure()

# Get unique source categories from DataFrame
unique_sources = sns_df2['출처분류명'].unique()

# Add bars for each source and keyword, initially hidden
for source in unique_sources:
    for keyword in sns_df2[sns_df2['출처분류명'] == source]['대표키워드명'].unique():
        filtered_df = sns_df2[(sns_df2['출처분류명'] == source) & (sns_df2['대표키워드명'] == keyword)]
        fig15.add_trace(
            go.Bar(
                x=[keyword],
                y=filtered_df['대표키워드언급수'],
                name=keyword,
                visible=False,  # initially all traces are hidden
                legendgroup=source,  # group by source for toggling
                legendgrouptitle_text=source  # show source as group title
            )
        )

# Create buttons for each source to toggle visibility
buttons = []

for i, source in enumerate(unique_sources):
    visibility = [(trace.legendgroup == source) for trace in fig15.data]  # adjust visibility based on group

    buttons.append(
        dict(
            label=source,
            method="update",
            args=[{"visible": visibility},
                  {"title": f"{source} - 대표키워드별 언급수"}]
        )
    )

# Add dropdown menu with buttons to the layout
fig15.update_layout(
    updatemenus=[dict(
        active=0,
        buttons=buttons,
        direction="down",
        pad={"r": 10, "t": 10},
        showactive=True,
        x=0.1,
        xanchor="left",
        y=1.2,
        yanchor="top"
    )],
    title=f"{unique_sources[0]} - 대표키워드별 언급수"
)

# Set the visibility of the first source category as default
for trace in fig15.data:
    trace.visible = trace.legendgroup == unique_sources[0]
#################################################################
fig16 = go.Figure()

# Get unique season categories from DataFrame
unique_sources = top_seasons['계절'].unique()

# Add bars for each season and location, initially hidden
for source in unique_sources:
    for keyword in top_seasons[top_seasons['계절'] == source]['지역명'].unique():
        filtered_df = top_seasons[(top_seasons['계절'] == source) & (top_seasons['지역명'] == keyword)]
        fig16.add_trace(
            go.Bar(
                x=[keyword],
                y=filtered_df['전체조회'],
                name=keyword,
                visible=False,  # initially all traces are hidden
                legendgroup=source,  # group by season for toggling
                legendgrouptitle_text=source  # show season as group title
            )
        )

# Create buttons for each season to toggle visibility
buttons = []

for i, source in enumerate(unique_sources):
    visibility = [(trace.legendgroup == source) for trace in fig16.data]  # adjust visibility based on group

    buttons.append(
        dict(
            label=source,
            method="update",
            args=[{"visible": visibility},
                  {"title": f"{source} - 계절별 검색어 언급수"}]
        )
    )

# Add dropdown menu with buttons to the layout
fig16.update_layout(
    updatemenus=[dict(
        active=0,
        buttons=buttons,
        direction="down",
        pad={"r": 10, "t": 10},
        showactive=True,
        x=0.1,
        xanchor="left",
        y=1.2,
        yanchor="top"
    )],
    title=f"{unique_sources[0]} - 계절별 검색어 언급수"
)

# Set the visibility of the first season's data as default
for trace in fig16.data:
    trace.visible = trace.legendgroup == unique_sources[0]
##################################################################
fig17 = go.Figure()

# Get unique classification names from DataFrame
unique_sources = top10_classification_df['분류명'].unique()

# Add bars for each classification and keyword, initially hidden
for source in unique_sources:
    for keyword in top10_classification_df[top10_classification_df['분류명'] == source]['지역명'].unique():
        filtered_df = top10_classification_df[(top10_classification_df['분류명'] == source) & (top10_classification_df['지역명'] == keyword)]
        fig17.add_trace(
            go.Bar(
                x=[keyword],
                y=filtered_df['전체조회'],
                name=keyword,
                visible=False,  # initially all traces are hidden
                legendgroup=source,  # group by classification for toggling
                legendgrouptitle_text=source  # show classification as group title
            )
        )

# Create buttons for each classification to toggle visibility
buttons = []

for i, source in enumerate(unique_sources):
    visibility = [(trace.legendgroup == source) for trace in fig17.data]  # adjust visibility based on group

    buttons.append(
        dict(
            label=source,
            method="update",
            args=[{"visible": visibility},
                  {"title": f"{source} - 분류별 검색어 언급수"}]
        )
    )

# Add dropdown menu with buttons to the layout
fig17.update_layout(
    updatemenus=[dict(
        active=0,
        buttons=buttons,
        direction="down",
        pad={"r": 10, "t": 10},
        showactive=True,
        x=0.1,
        xanchor="left",
        y=1.2,
        yanchor="top"
    )],
    title=f"{unique_sources[0]} - 분류별 검색어 언급수"
)

# Set the visibility of the first classification's data as default
for trace in fig17.data:
    trace.visible = trace.legendgroup == unique_sources[0]
######################################################################
region_consumption_sorted1 = load_data("Jeju/시각화/region_consumption_sorted1.csv")
region_variation_sorted = load_data("Jeju/시각화/region_variation_sorted.csv")
top_local_sales_cleaned = load_data("Jeju/시각화/top_local_sales_cleaned.csv")
top_foreign_sales_cleaned = load_data("Jeju/시각화/top_foreign_sales_cleaned.csv")
sorted_grouped_df = load_data("Jeju/시각화/sorted_grouped_df.csv")
time_df = load_data("Jeju/시각화/time_df.csv")
#######################################################################
region_consumption_sorted1['년'] = region_consumption_sorted1['년'].astype(str)

fig18 = go.Figure()

# Create a list of unique '년' (years)
unique_years = region_consumption_sorted1['년'].unique()

# Add a trace for each year and region
for year in unique_years:
    for region in region_consumption_sorted1[region_consumption_sorted1['년'] == year]['지역명'].unique():
        filtered_df = region_consumption_sorted1[(region_consumption_sorted1['년'] == year) & (region_consumption_sorted1['지역명'] == region)]
        fig18.add_trace(
            go.Bar(
                x=filtered_df['지역명'],
                y=filtered_df['전체매출금액비율'],
                name=f"{year} - {region}",
                visible=False, # initially all traces are hidden
                customdata=[year] * len(filtered_df)
            )
        )

# Create buttons for each year to toggle visibility
buttons = []

for i, year in enumerate(unique_years):
    visibility = [year == trace.customdata[0] for trace in fig18.data]
    buttons.append(
        dict(
            label=year,
            method="update",
            args=[{"visible": visibility},
                  {"title": f"{year}년 지역별 전체매출금액비율"}]
        )
    )

# Update layout with dropdown menu for the buttons
fig18.update_layout(
    updatemenus=[{
        "buttons": buttons,
        "direction": "down",
        "pad": {"r": 10, "t": 10},
        "showactive": True,
        "x": 0.1,
        "xanchor": "left",
        "y": 1.2,
        "yanchor": "top"
    }],
    title=f"{unique_years[0]}년 지역별 전체매출금액비율"
)

# Set initial visibility for the first year
initial_year = unique_years[0]
for trace in fig18.data:
    trace.visible = trace.customdata[0] == initial_year
###################################################
region_variation_sorted['년'] = region_variation_sorted['년'].astype(str)

fig19 = go.Figure()

# Create a list of unique years
unique_years = region_variation_sorted['년'].unique()

# Add a bar for each year and region, initially hidden
for year in unique_years:
    for region in region_variation_sorted[region_variation_sorted['년'] == year]['지역명'].unique():
        filtered_df = region_variation_sorted[(region_variation_sorted['년'] == year) & (region_variation_sorted['지역명'] == region)]
        fig19.add_trace(
            go.Bar(
                x=filtered_df['지역명'],
                y=filtered_df['변화율'],
                name=f"{year} - {region}",
                visible=False,  # initially all traces are hidden
                customdata=[year] * len(filtered_df)
            )
        )

# Create buttons for interactivity
buttons = []

for i, year in enumerate(unique_years):
    visibility = [year == trace.customdata[0] for trace in fig19.data]
    buttons.append(
        dict(
            label=year,
            method="update",
            args=[{"visible": visibility},
                  {"title": f"{year}년 지역별 변화율"}]
        )
    )

# Apply updated button logic
fig19.update_layout(
    updatemenus=[{
        "buttons": buttons,
        "direction": "down",
        "pad": {"r": 10, "t": 10},
        "showactive": True,
        "x": 0.1,
        "xanchor": "left",
        "y": 1.2,
        "yanchor": "top"
    }],
    title=f"{unique_years[0]}년 지역별 변화율"
)

# Set initial visibility based on the first year
initial_year = unique_years[0]
for trace in fig19.data:
    trace.visible = trace.customdata[0] == initial_year
###########################################################
top_local_sales_cleaned['년'] = top_local_sales_cleaned['년'].astype(str)

fig20 = go.Figure()

# Create a list of unique years
unique_years = top_local_sales_cleaned['년'].unique()

# Add a bar for each year and business name, initially hidden
for year in unique_years:
    for business in top_local_sales_cleaned[top_local_sales_cleaned['년'] == year]['상호명'].unique():
        filtered_df = top_local_sales_cleaned[(top_local_sales_cleaned['년'] == year) & (top_local_sales_cleaned['상호명'] == business)]
        fig20.add_trace(
            go.Bar(
                x=[business],  # x-axis is the business name
                y=filtered_df['제주도민매출금액비율'],  # y-axis is the sales ratio
                name=business,
                visible=False,  # initially all traces are hidden
                customdata=[year] * len(filtered_df)
            )
        )

# Create buttons for interactivity
buttons = []

for i, year in enumerate(unique_years):
    visibility = [year == trace.customdata[0] for trace in fig20.data]
    buttons.append(
        dict(
            label=year,
            method="update",
            args=[{"visible": visibility},
                  {"title": f"{year}년 관광지별 제주도민매출금액비율"}]
        )
    )

# Apply updated button logic
fig20.update_layout(
    updatemenus=[{
        "buttons": buttons,
        "direction": "down",
        "pad": {"r": 10, "t": 10},
        "showactive": True,
        "x": 0.1,
        "xanchor": "left",
        "y": 1.2,
        "yanchor": "top"
    }],
    title=f"{unique_years[0]}년 관광지별 제주도민매출금액비율"
)

# Set initial visibility based on the first year
initial_year = unique_years[0]
for trace in fig20.data:
    trace.visible = trace.customdata[0] == initial_year
##############################################################
top_foreign_sales_cleaned['년'] = top_foreign_sales_cleaned['년'].astype(str)

fig21 = go.Figure()

# Create a list of unique years
unique_years = top_foreign_sales_cleaned['년'].unique()

# Add a bar for each year and business name, initially hidden
for year in unique_years:
    for business in top_foreign_sales_cleaned[top_foreign_sales_cleaned['년'] == year]['상호명'].unique():
        filtered_df = top_foreign_sales_cleaned[(top_foreign_sales_cleaned['년'] == year) & (top_foreign_sales_cleaned['상호명'] == business)]
        fig21.add_trace(
            go.Bar(
                x=[business],  # x-axis is the business name
                y=filtered_df['외지인매출금액비율'],  # y-axis is the non-resident sales ratio
                name=business,
                visible=False,  # initially all traces are hidden
                customdata=[year] * len(filtered_df)
            )
        )

# Create buttons for interactivity
buttons = []

for i, year in enumerate(unique_years):
    visibility = [year == trace.customdata[0] for trace in fig21.data]
    buttons.append(
        dict(
            label=year,
            method="update",
            args=[{"visible": visibility},
                  {"title": f"{year}년 상호별 외지인매출금액비율"}]
        )
    )

# Apply updated button logic
fig21.update_layout(
    updatemenus=[{
        "buttons": buttons,
        "direction": "down",
        "pad": {"r": 10, "t": 10},
        "showactive": True,
        "x": 0.1,
        "xanchor": "left",
        "y": 1.2,
        "yanchor": "top"
    }],
    title=f"{unique_years[0]}년 상호별 외지인매출금액비율"
)

# Set initial visibility based on the first year
initial_year = unique_years[0]
for trace in fig21.data:
    trace.visible = trace.customdata[0] == initial_year
###############################################
fig22 = px.scatter(
    sorted_grouped_df,
    x="전체매출금액비율",
    y="전체매출수비율",
    animation_frame="지역명",
    animation_group="소분류명",
    size="전체매출금액비율",
    color="소분류명",
    hover_name="소분류명",
    log_x=True,
    log_y=True,
    size_max=55,
    range_x=[0.01, 12],
    range_y=[0.005, 65]
)

# Remove animation play and pause buttons
fig22["layout"].pop("updatemenus")
##############################################
fig23 = go.Figure()

# Get unique categories from DataFrame
categories = time_df['중분류명'].unique()

# Add data for each category to the graph
for category in categories:
    category_df = time_df[time_df['중분류명'] == category]
    fig23.add_trace(go.Scatter(x=category_df['분석년월'], y=category_df['외지인매출금액비율'], mode='lines+markers', name=category))

# Set the title
fig23.update_layout(title_text="Time series of non-resident sales ratio by category")

# Add range slider
fig23.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    )
)
###########################################################################
df_pca_2d = load_data("Jeju/군집분석/PCA_2d.csv")
df_pca_3d = load_data("Jeju/군집분석/PCA_3d.csv")
############################################################################
# 2차원 PCA 시각화
fig_pca_2d = px.scatter(
    df_pca_2d,
    x='PC1',
    y='PC2',
    hover_name='숙박업명',  # 숙박업명 컬럼을 호버 텍스트로 사용
    title='2D PCA of Accommodation Facility Data'
)
#############################################################################
# 3차원 PCA 시각화
fig_pca_3d = px.scatter_3d(df_pca_3d, x='PC1', y='PC2', z='PC3', hover_name='숙박업명', title='3D PCA of Accommodation Facility Data')
fig_pca_3d.update_traces(marker=dict(size=3))  # 모든 점의 크기를 조정합니다.
#############################################################################
df_elbow1 = load_data("Jeju/군집분석/df_elbow.csv")
df_silhouette1 = load_data("Jeju/군집분석/df_silhouette.csv")
#################################################################################
# 엘보우 방법 시각화 (df_elbow 데이터프레임 사용)
fig_elbow1 = px.line(
    df_elbow1,
    x='Number of Clusters',
    y='Inertia',
    markers=True,
    labels={'x': 'Number of Clusters', 'y': 'Inertia'}
)
fig_elbow1.update_layout(title="Elbow Method with Plotly")

# 실루엣 분석 시각화 (df_silhouette 데이터프레임 사용)
fig_silhouette1 = px.line(
    df_silhouette1,
    x='Number of Clusters',
    y='Silhouette Score',
    markers=True,
    labels={'x': 'Number of Clusters', 'y': 'Silhouette Score'}
)
fig_silhouette1.update_layout(title="Silhouette Analysis with Plotly")
#####################################################################
clustered_pca_data = load_data("Jeju/군집분석/clustered_pca_data.csv")
##########################################################################
# 실루엣 점수 시각화
fig24 = go.Figure()
fig24.add_trace(go.Bar(x=clustered_pca_data['Model'], y=clustered_pca_data['Silhouette Score']))
fig24.update_layout(title="Silhouette Scores for Different Clustering Models",
                  xaxis_title="Model",
                  yaxis_title="Silhouette Score")
#######################################################################
df_umap_2d = load_data("Jeju/군집분석/df_umap_2d.csv")
df_umap_3d = load_data("Jeju/군집분석/df_umap_3d.csv")
#######################################################################
fig_umap_2d = px.scatter(
    df_umap_2d,
    x='UMAP1',
    y='UMAP2',
    hover_name='숙박업명',  # 숙박업명 컬럼을 호버 텍스트로 사용
    title='2D UMAP of Accommodation Facility Data'
)
################################################################
fig_umap_3d = px.scatter_3d(df_umap_3d, x='UMAP1', y='UMAP2', z='UMAP3', hover_name='숙박업명', title='3D UMAP of Accommodation Facility Data')
fig_umap_3d.update_traces(marker=dict(size=3))  # 모든 점의 크기를 조정합니다.
#########################################################################
df_elbow_umap = load_data("Jeju/군집분석/df_elbow_umap.csv")
df_silhouette_umap = load_data("Jeju/군집분석/df_silhouette_umap.csv")
###############################################################################
# 엘보우 방법 시각화 (df_elbow 데이터프레임 사용)
fig_elbow2 = px.line(
    df_elbow_umap,
    x='Number of Clusters',
    y='Inertia',
    markers=True,
    labels={'x': 'Number of Clusters', 'y': 'Inertia'}
)
fig_elbow2.update_layout(title="Elbow Method with Plotly")

# 실루엣 분석 시각화 (df_silhouette 데이터프레임 사용)
fig_silhouette2 = px.line(
    df_silhouette_umap,
    x='Number of Clusters',
    y='Silhouette Score',
    markers=True,
    labels={'x': 'Number of Clusters', 'y': 'Silhouette Score'}
)
fig_silhouette2.update_layout(title="Silhouette Analysis with Plotly")
########################################################################
clustered_umap_data = load_data("Jeju/군집분석/clustered_umap_data.csv")
##########################################################################
# 실루엣 점수 시각화
fig25 = go.Figure()
fig25.add_trace(go.Bar(x=clustered_umap_data['Model'], y=clustered_umap_data['Silhouette Score']))
fig25.update_layout(title="Silhouette Scores for Different Clustering Models",
                  xaxis_title="Model",
                  yaxis_title="Silhouette Score")
########################################################################
merged_df = load_data("Jeju/군집분석/군집_최종.csv")
####################################################################3
clusters = merged_df['kmeans_cluster'].unique()
trace_data = {}
for cluster in clusters:
    cluster_data = merged_df[merged_df['kmeans_cluster'] == cluster]
    trace_data[cluster] = {
        'x': cluster_data['UMAP1'],
        'y': cluster_data['UMAP3'],
        'text': cluster_data['숙박업명']  # 숙박업명을 text 데이터로 추가
    }

fig26 = go.Figure()

# 색상 팔레트 생성
colors = px.colors.qualitative.Plotly  # 더 많은 색상이 필요하면 다른 팔레트 선택

# 각 군집별로 트레이스 추가
for idx, cluster in enumerate(clusters):
    color_index = idx % len(colors)  # 색상 반복 사용
    fig26.add_trace(
        go.Scatter(
            x=trace_data[cluster]['x'],
            y=trace_data[cluster]['y'],
            mode="markers",
            marker=dict(color=colors[color_index]),
            name=f"Cluster {cluster}",
            text=trace_data[cluster]['text'],  # 호버 텍스트로 숙박업명 사용
            hoverinfo='text'  # 호버 정보를 텍스트로 제한
        )
    )

# 군집별 원형 표시를 위한 동적 버튼 추가
cluster_shapes = []
for idx, cluster in enumerate(clusters):
    x0, y0 = min(trace_data[cluster]['x']), min(trace_data[cluster]['y'])
    x1, y1 = max(trace_data[cluster]['x']), max(trace_data[cluster]['y'])
    color_index = idx % len(colors)  # 색상 반복 사용
    cluster_shapes.append(dict(type="circle",
                               xref="x", yref="y",
                               x0=x0, y0=y0,
                               x1=x1, y1=y1,
                               line=dict(color=colors[color_index])))

buttons = []
for idx, cluster in enumerate(clusters):
    buttons.append(
        dict(label=f"Cluster {cluster}",
             method="relayout",
             args=["shapes", [cluster_shapes[idx]]])
    )

buttons.append(
    dict(label="All",
         method="relayout",
         args=["shapes", cluster_shapes])
)

fig26.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            buttons=buttons
        )
    ],
    title_text="Highlight Clusters with 2D UMAP",
    showlegend=True
)
###############################################
combined_df = load_data("Jeju/데이터/종합맵.csv")
################################################
def create_map(df):
    # 제주도의 중심 좌표
    jeju_center = [33.3617, 126.5292]

    # Folium을 사용하여 지도 생성
    map_jeju = folium.Map(location=jeju_center, zoom_start=10)

    # 분류명별로 마커 색상과 모양을 지정 (임의 지정)
    marker_colors = {
        '반려견 동반 관광지': 'blue',
        '마을 관광자원': 'green',
        '안전여행 스탬프 관광지': 'red'
    }

    # MarkerCluster 객체 생성
    marker_cluster = MarkerCluster().add_to(map_jeju)

    # 데이터에 따라 마커 추가
    for idx, row in df.iterrows():
        color = marker_colors.get(row['분류'], 'gray')  # 색상 선택
        popup_text = f"<strong>{row['관광지명']}</strong><br>" \
                     f"{row['관광지분류']}<br>" \
                     f"{row['주소']}<br>" \
                     f"{row['관광지설명']}"

        # 마커 아이콘 설정
        icon = folium.Icon(color=color)
        
        # MarkerCluster에 마커 추가
        folium.Marker(
            location=[row['위도'], row['경도']],
            popup=folium.Popup(popup_text, max_width=300),
            icon=icon
        ).add_to(marker_cluster)

    return map_jeju
#######################################
# rest_1 = load_data("Jeju/제주주변식당/HW_JJ_LDGS_CFR_RSTRNT_PREFEER_INFO_202303.csv")
# rest_2 = load_data("Jeju/제주주변식당/HW_JJ_LDGS_CFR_RSTRNT_PREFEER_INFO_202306.csv")
# rest_3 = load_data("Jeju/제주주변식당/HW_JJ_LDGS_CFR_RSTRNT_PREFEER_INFO_202309.csv")
###################################
jeju_downtown_review = pd.read_csv('Jeju/네이버리뷰_크롤링/jeju_downtown_review.csv', index_col=0)
jeju_city_review = pd.read_csv('Jeju/네이버리뷰_크롤링/jeju_city_review.csv', index_col=0)
###################################
total_keyword = load_data('Jeju/전처리데이터셋/제주숙박리뷰키워드(a).csv')
review_explode = load_data('Jeju/전처리데이터셋/제주숙박리뷰키워드(b).csv')
###################################
keyword_final = load_data('Jeju/전처리데이터셋/final_keyword.csv')
##################################
final_accomodation_recommendation = load_data('Jeju/전처리데이터셋/final_hotel_recommendation.csv')
############################################
def map_lodge(df):
  lodge_map = folium.Map(location=[33.3617, 126.5332], zoom_start=10)


  for index, row in final_accomodation_recommendation.iterrows():
      location = [row['위도'], row['경도']]
      popup = folium.Popup(f"<b style='font-size: 16px;'>{row['숙박업명']}</b>", max_width=300) # </b>~</b> 글씨 진하게
      folium.Marker(location=location, popup=popup).add_to(lodge_map)
      
  return lodge_map
###################################
final_food_df = load_data('Jeju/전처리데이터셋/제주_검색량_거리.csv')

# 호텔별 최단거리 / 검색량 최고 호텔
def restaurant_map(df_1, df_2):
    m = folium.Map(location=[33.3617, 126.5332], zoom_start=10)
    marker_cluster = MarkerCluster().add_to(m)

    # 숙박 업소
    for index, row in df_1.iterrows():
        location = [row['위도'], row['경도']]
        popup = folium.Popup(f"<b style='font-size: 16px;'>{row['숙박업명']}</b>", max_width=300)
        folium.Marker(location=location, popup=popup, icon=folium.Icon(color='purple')).add_to(m)

        folium.Circle(location=location, radius=3000, color='gray', fill=True, fill_color='gray').add_to(m)

        # 숙박 업소와 가장 가까운 식당 찾기
        min_distance = float('inf')
        closest_restaurant_loc = None
        for _, restaurant_row in df_2.iterrows():
            restaurant_loc = [restaurant_row['식당위도'], restaurant_row['식당경도']]
            distance = haversine(location, restaurant_loc)
            if distance < min_distance:
                min_distance = distance
                closest_restaurant_loc = restaurant_loc

        # 숙박 업소와 가장 가까운 식당의 마커와 연결선 그리기
        if closest_restaurant_loc:
            popup = folium.Popup(f"<b style='font-size: 16px;'>{restaurant_row['식당명']}</b>", max_width=300)
            folium.Marker(location=closest_restaurant_loc,
                          popup=popup,
                          icon=folium.Icon(color='blue')).add_to(m)
            folium.PolyLine(locations=[location, closest_restaurant_loc], color='blue').add_to(m)

        # 해당 숙박 업소에 대한 검색량이 가장 높은 식당 찾기
        accomodation_name = row['숙박업명']
        most_searched_restaurant = df_2[df_2['숙박업명'] == accomodation_name].iloc[0]
        most_searched_restaurant_loc = [most_searched_restaurant['식당위도'], most_searched_restaurant['식당경도']]

        # 숙박 업소와 가장 검색량이 높은 식당의 마커와 연결선 그리기
        popup= folium.Popup(f"<b style='font-size: 16px;'>{most_searched_restaurant['식당명']}</b>", max_width=300)
        folium.Marker(location=most_searched_restaurant_loc,
                      popup=popup,
                      icon=folium.Icon(color='lightred')).add_to(m)
        folium.PolyLine(locations=[location, most_searched_restaurant_loc], color='red').add_to(m)

    # 군집화할 나머지 식당
    for index, row in df_2.iterrows():
        location = [row['식당위도'], row['식당경도']]
        popup = folium.Popup(f"<b style='font-size: 16px;'>{row['식당명']}</b>", max_width=300)
        folium.Marker(location=location, popup=popup, icon=None).add_to(marker_cluster)
    
    return m
  
###################################
restaurant_info_df = load_data('Jeju/전처리데이터셋/숙박업별_최단거리_최다검색.csv')

fig_distance = go.Figure()

fig_distance.add_trace(go.Bar(
    x=restaurant_info_df['최단거리'],
    y=restaurant_info_df['숙박업명'],
    text=restaurant_info_df['가장가까운식당'],  # 막대 위에 텍스트 추가
    textposition='inside',  # 텍스트 위치 설정
    name='Closest Restaurant',
    orientation='h',  # 수평 막대 그래프
    marker=dict(color='skyblue'),  # 막대 색상 지정
))

fig_distance.update_layout(
    title='숙박업별 최단 거리 추천식당',
    xaxis=dict(title='거리 (km)'),
    yaxis=dict(title='숙박업명'),
    bargap=0.1,  # 막대 간 간격 조정
)

# 검색량을 나타내는 그래프
fig_search_count = go.Figure()

fig_search_count.add_trace(go.Bar(
    x=restaurant_info_df['최고검색량'],
    y=restaurant_info_df['숙박업명'],
    text=restaurant_info_df['가장높은검색량식당'],  # 막대 위에 텍스트 추가
    textposition='inside',  # 텍스트 위치 설정
    name='Most Searched Restaurant',
    orientation='h',  # 수평 막대 그래프
    marker=dict(color='lightgreen'),  # 막대 색상 지정
))

fig_search_count.update_layout(
    title='숙박업별 최다 검색량 추천식당',
    xaxis=dict(title='검색량 합계값'),
    yaxis=dict(title='숙박업명'),
    bargap=0.1,
)  # 막대 간 간격 조정
############################################
###################################
wordcloud_pos_review = Image.open("Jeju/숙박시설리뷰감정분석/워드클라우드_제주리뷰키워드.png")
wordcloud_city_keyword = Image.open("Jeju/숙박시설리뷰감정분석/wordcloud_city_keyword.png")
wordcloud_downtown_keyword = Image.open("Jeju/숙박시설리뷰감정분석/wordcloud_downtown_keyword.png")
###################################
final_city_review = pd.read_csv('Jeju/전처리데이터셋/final_city_review.csv', index_col=0)
final_downtown_review = pd.read_csv('Jeju/전처리데이터셋/final_downtown_review.csv', index_col=0)
###############################################

# 리뷰 가중치 점수 part
unique_values = total_keyword['ratio_weight']

fig27 = go.Figure(data=[go.Histogram(x=unique_values, marker_color='skyblue', opacity=0.7)])

fig27.update_layout(
    title='숙박 키워드별 가중치 점수 분포',
    xaxis_title='가중치 부여 점수',
    yaxis_title='총계',
    bargap=0.05,  # 막대 간격 조절
    bargroupgap=0.1,  # 그룹 간격 조절
    plot_bgcolor='rgba(0,0,0,0)',  # 배경색 투명도 설정
    xaxis=dict(tickmode='linear', tick0=0, dtick=0.01),
    yaxis=dict(tickmode='linear', tick0=0, dtick=100)
)

############################################################
    # ratio_weight 값으로 내림차순 정렬
keyword_final_sorted = keyword_final.sort_values(by='ratio_weight', ascending=False)

    # 점수 범위에 따라 색상 설정
colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)']
keyword_final_sorted['color'] = pd.cut(keyword_final_sorted['ratio_weight'],
                                       bins=[0, 9.99, 10, 10.99, float('inf')],
                                       labels=colors,
                                       right=False)

fig31 = go.Figure()

fig31.add_trace(go.Bar(
    x=keyword_final_sorted.index,
    y=keyword_final_sorted['ratio_weight'],
    marker=dict(
        color=keyword_final_sorted['color']
    ),
    text=keyword_final_sorted['ratio_weight'].apply(lambda x: f"{x:.3f}"),  # 막대 가운데에 소수점 3자리까지 표시
    textposition='auto',  # 텍스트 위치 설정 (auto: 자동으로 가장 적절한 위치에 표시)
))

fig31.update_layout(
    title='  가중치 점수 반영된 최종 40개 숙박업의 점수',
    xaxis=dict(title='숙박업명'),
    yaxis=dict(title='ratio_weight'),
)
##########################################################################3
# 제주시/서귀포시 상위 5개 호텔 점수
grouped_df = final_accomodation_recommendation.groupby('구역')

fig32 = go.Figure()

for area, area_df in grouped_df:
    fig32.add_trace(go.Bar(
        x=area_df['숙박업명'],
        y=area_df['ratio_weight'],
        name=area,
        text=round(area_df['ratio_weight'],3),  # 막대 가운데에 표시할 값
        textposition='auto',  # 텍스트 위치 설정 (auto: 자동으로 가장 적절한 위치에 표시)
    ))

fig32.update_layout(
    title='  제주시/서귀포시 점수 상위 5개 호텔',
    xaxis=dict(title='숙박업명'),
    yaxis=dict(title='ratio_weight'),
    barmode='group'
)




























# 페이지 및 섹션 정의
pages = [
    Page("개요", 
         """
         ### 프로젝트 목표
         - 제주도는 자연의 아름다움과 독특한 문화로 유명한 관광지입니다. 이 프로젝트는 방대한 제주도 관광 데이터를 분석하여 관광 산업의 현황을 파악하고, 관광 산업의 발전 가능성을 탐색하기 위해 시작되었습니다. 관광 데이터를 통해 도출된 인사이트는 관광 정책 결정자, 사업자 및 방문객에게 유용한 정보를 제공할 것입니다.
         
         
         ### 데이터 소스
         - 분석에 사용된 데이터는 제주 관광공사, KDX 한국데이터거래소, 제주데이터허브에서 수집한 다양한 유형의 데이터를 포함합니다. 이들 데이터에는 동반자 유형별 여행 계획, 무장애 관광지 입장 데이터, 소비 행태 데이터, SNS에서 수집한 관광 키워드 데이터 등이 포함됩니다.
         
         
         ### 분석 구성
         - 시계열 분석: 향후 제주도 경제 상황 예측을 위해 소비를 시간에 따라 분석합니다.
         - 관광 현황 분석: 동반자 유형별로 관광 현황과 선호도를 분석합니다.
         - 키워드 분석: SNS 데이터를 활용하여 제주도에 대한 대중의 관심사를 파악합니다.
         - 소비 행태 분석: 지역별, 분류별로 관광객의 소비 행태를 분석합니다.
         - 호텔 군집 모델링: 숙박시설 데이터를 활용해 시장을 세분화하고 군집을 형성합니다.
         - 추천 시스템: 분석 결과를 바탕으로 관광객에게 맞춤형 추천 식당을 제공합니다.
         - 추천 관광지: 종류별 추천 관광지를 지도화하여 제공합니다.
         
         ### 사용할 도구와 기술
         - 데이터 분석 및 모델링에는 Python을 주 언어로 사용하며, Pandas, Plotly, Folium, Streamlit 등의 라이브러리와 프레임워크를 활용했습니다. 시계열 분석에는 Prophet 모델을 적용했습니다.
         
         """
    ),
    Page("데이터 소개", 
         """
         ### 데이터 샘플 
         
         """,
         dfs=[df_1, df_2, df_3, df_4, df_5, df_6, df_7],
         df_titles=["제주 동반자 유형별 여행 계획 데이터", "제주 무장애 관광지 입장 데이터",
                    "SNS 제주 관광 키워드별 수집 통계_월", "제주 관광수요예측 데이터_비짓제주 로그 데이터",
                    "제주관광공사 관광 소비행태 데이터 카드사 음식 급상승 데이터", "제주관광공사 관광 소비행태 데이터 카드사 음식 급상승 데이터(21~23)_수정",
                    "[NH농협카드] 일자별 소비현황_제주", "제주도 맵 데이터(관광자원, 반려경 동반 관광지, 안전여행 스탬프 관광지)"]

    ),
    Page("농협카드 - 데이터 확인", 
         """
         ## 계절성, 추세, 정상성, 노이즈 분석
         """,
         graphs=[fig1, fig2, fig3, fig4],
         graph_descriptions=[
             "정기적인 간격으로 반복되는 패턴이 뚜렷이 나타나 계절성이 명확히 확인.",
             "이동 평균을 활용한 그래프 분석 결과, 점차 상승하는 추세가 확인.",
             "첫 번째 래그(lag)에서 1의 값을 가지고 이후 7을 주기로 상관관계가 점차 감소하고있기에 어느정도 정상성을 가진다고 판단.",
             "붉은색으로 표시된 평활화된 데이터가 파란색의 원본 데이터에 대해 어떤 일관된 추세를 보여주기에 큰 노이즈가 없다고 판단.",
         ]
         
    ),
    Page("제주도의 미래 소비 예측을 위한 Prophet모델링", 
         """
         ### 계절성이 있는 데이터에 적합한 모델인 Prophet 선택
         """,
         graphs=[fig5, components_fig5, fig6, components_fig6],
         graph_descriptions=[
             "RMSE: 870 / 성능 향상을 위해 다양한 방법을 고안하였습니다.",
             "holidays 변수, 변화점 조정, 하이퍼파라미터 튜닝을 통해 모델의 성능 향상 도모.",
             "RMSE: 579로 모델 성능의 향상을 끝마쳤습니다.",
             "분석 결과 23년도 1월 1일 대비 2024년 1월 1일에 4.28%의 소비가 감소한 것을 확인할 수 있습니다."
         ]
    ), 
    Page("동반자 유형 & 관광지 분석", 
         """
         ### 관광 현황 분석

         """,
         graphs=[fig7, fig8, fig10, fig11],
         graph_descriptions=[
             "1월부터 9월까지는 가족단위 관광객이 50% 이상인 반면, 9월부터 12월까지는 가족단위 관광객이 25%로 하락하고  친구와 함께 방문한 관광객이 20%내외에서 41%로 증가하였습니다.",
             "대부분의 유형에서 '휴식과 치유 여행'이 50%가 넘는 비율을 차지하고 있습니다. 하지만 아이와 함께한 관광객은 42.6%로 상대적으로 낮고 레저와 체험이 24.9%로 높은 것을 확인할 수 있습니다.",
             "천지연폭포는 다른 폭포들과 달리, 4월 대비 7월에 관광객이 62.43%나 급감한 것을 확인할 수 있습니다.",
             "경로, 유아, 장애인 방문객 모두 폭포를 가장 많이 방문한 것으로 확인되었습니다."
         ]
         
    ),    
    Page("SNS를 활용한 키워드 분석", 
         """
         ### 키워드를 활용한 관광 분석
         """,
         graphs=[fig13, fig15, fig16, fig17],
         graph_descriptions=[
             "2022년 상반기 대비 산방산 맛집에 대한 언급이 1917.8% 증가한 것을 확인할 수 있습니다. 또한 2022년에는 관광 명소에 대한 관심이 많았으며, 2023년에는 미식 장소에 대한 관심이 크게 상승했습니다.",
             "네이버 블로그에서는 '맛집' 키워드가 인기를 끌었고, 인스타그램과 페이스북에서는 관광지에 대한 언급이 많았습니다. 트위터에서는 '제주도렌트카'키워드의 양이 가장 많은 것이 두드러지는 특징입니다.",
             "계절에 따라 약간의 차이는 있지만, 사려니숲길, 성산일출봉, 비자림, 우도는 일관되게 높은 검색량을 기록하고 있는 것을 확인할 수 있습니다.",
             "각 분류별로 언급량이 가장 많은 10개의 장소입니다."
         ]
    ),
    Page("음식 소비행태 분석",
         """
         ### 신한카드 데이터를 활용한 음식 소비행태 분석
        
         """,
         graphs=[fig18, fig19, fig20, fig21, fig22, fig23],
         graph_descriptions=[
             "안덕면에서 가장 큰 매출을 기록하고 있으며, 시간이 지남에 따라 서귀포 시내와 제주 시내의 매출이 상승하고 있습니다.",
             "안덕면, 조천읍, 애월읍에서 큰 변화를 보이고 있으며, 제주 시내의 변화율도 점차 증가하고 있습니다.",
             "제주도민에서는 돼지고기 관련 매출이 눈에 띄게 높은 것을 확인할 수 있습니다.",
             "외지인들은 다양한 식당에서의 소비가 확인되고 있습니다.",
             "지역별 현황을 확인할 수 있습니다. 안덕면에서는 특이하게 '차'관련 매출이 가장 높은 것을 확인할 수 있습니다.",
             "1월부터 3월까지는 매출이 상승하고 있고 4월부터 8월까지는 매출이 급감하다가 9월부터 11월까지 다시 매출이 상승하고 12월에 다시 떨어지는 패턴을 보이고 있습니다.."
         ]         
    ),
    Page("숙박시설 - 제주호텔 군집분석",
         """
         ### 스케일링 & 인코딩
         """            
    ),
    Page("숙박시설 - 차원축소(PCA)",
         """
         ### 실루엣 계수를 통한 PCA 적합성 확인
         """,
         graphs = [fig_pca_2d, fig_pca_3d, fig_elbow1, fig_silhouette1, fig24],
         graph_descriptions=[
             "PCA를 활용한 2차원 차원축소 결과입니다.",
             "PCA를 활용한 3차원 차원축소 결과입니다.",
             "엘보우 방법 결과 3개의 군집으로 나누는 것이 좋은 결과로 보입니다.",
             "실루엣 계수 확인 결과 3개의 군집으로 나누는 것이 좋은 결과로 보입니다.",
             "3개의 모델로 확인해본 실루엣 계수가 0.3으로 낮아서 다른 차원축소 방법이 좋아보입니다."
         ]           
    ),
    Page("숙박시설 - 차원축소(UMAP)",
         """
         ### 실루엣 계수를 통한 UMAP  적합성 확인
         """,
         graphs=[fig_umap_2d, fig_umap_3d, fig_elbow2, fig_silhouette2, fig25],
         graph_descriptions=[
             "UMAP을 활용한 2차원 차원축소 결과입니다.",
             "UMAP을 활용한 3차원 차원축소 결과입니다.",
             "엘보우 방법으로는 4개의 군집으로 나누는 것이 좋은 결과로 보입니다.",
             "실루엣 계수가 6에서 가장 높기에 6개의 군집으로 나누는 것이 좋은 결과로 보입니다..",
             "UMAP을 통한 차원축소 후 6개의 군집으로 나눈 결과 실루엣 계수가 0.5에 가깝게 상승하였습니다."
         ]         
    ),
    Page("숙박시설 - UMAP & K-Means",
         """
         ### 군집분석 활용한 호텔 분석
         """    
    ),
    Page("숙박 리뷰 키워드_호텔 점수 산정",
         """
    
         """,
         graphs=[fig27, fig31,fig32],
         graph_descriptions=[
             "각 키워드의 출현 빈도를 전체 키워드의 출현 총계로 나누어서, 각 키워드에 대한 점수에 빈도 비율에 해당하는 가중치를 부여한 점수 분포",
             "리뷰를 가진 제주도 호텔 40곳을 뽑아, 가중치 점수를 반영하여 각 호텔별 키워드 점수를 산출한 통계 ",
             "그 중 제주시/서귀포시 두 구역으로 나누어 점수가 높은 5곳의 호텔을 각각 선정"
         ]
    ),
    Page("지역별 상위 5개 호텔 & 식당 분포",
         """
         """,
         graphs=[(map_lodge(final_accomodation_recommendation),"선정된 10개의 숙박업소 위치"),
                 (restaurant_map(final_accomodation_recommendation, final_food_df), "호텔별 최단거리 식당과 최다 검색량 식당 위치"),
                 fig_distance, fig_search_count],
         graph_descriptions=["리뷰 기반 점수 시별 상위 5곳 호텔",
                             "거리/검색량 기반 호텔별 식당 추천",
                             "서귀포시는 제주시에 비해 추천식당이 비교적 거리가 있다."] # 왜 안나오는가
         
    ),
    Page("네이버 식당 리뷰 분석 (크롤링 및 자연어처리)",
         """
         """,
         dfs=[jeju_city_review, jeju_downtown_review, final_city_review, final_downtown_review],
         df_titles=['제주시 식당 리뷰 크롤링', 
                    '서귀포시 식당 리뷰 크롤링',
                    '자연어 처리 후 토큰화 최종 키워드(제주시)',
                    '자연어 처리 후 토큰화 최종 키워드(서귀포시)'
        ],
         image_title=['제주시 리뷰 키워드', '서귀포시 리뷰 키워드'],
         images=[wordcloud_city_keyword,wordcloud_downtown_keyword]
    ),
    Page("식당 추천시스템_제주시",
         """
         """),
    Page("식당 추천시스템_서귀포시",
         """
         """
    ),    
    Page("분류별 추천 관광지", 
         """
         ## 마을 광광자원, 반려견 동반 관광지, 안전여행 스탬프
         """,
         graphs=[create_map(combined_df)]
    ),
    Page("끝이 다가오는 것은 시작을 알리는 신호입니다", 
         """
         ## 
         """
    ),    
    
    
    
    
]


# 페이지 제목 추가
add_page_title()

# 왼쪽 사이드바에 페이지 목록 추가
selected_page = st.sidebar.radio("목차", [page.title for page in pages])

# 선택된 페이지로 이동
for page in pages:
    if page.title == selected_page:
        show_pages([page])




















