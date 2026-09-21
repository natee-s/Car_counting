import streamlit as st
import pandas as pd
import plotly.express as px
import cv2
from utils.yolo_tracker import VehicleTracker
from utils.stream_handler import VideoStreamer

# 1. Page Configuration
st.set_page_config(page_title="Real-time Traffic Analytics", layout="wide")

# 2. Custom CSS สำหรับปรับแต่ง Card สีสันสดใส
st.markdown("""
<style>
    /* ปรับแต่งภาพรวมของแอพให้ดูสะอาดขึ้น */
    .stApp { background-color: #f8f9fc; }
    
    /* สไตล์หลักของ Card */
    .dashboard-card {
        border-radius: 12px;
        padding: 20px 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        text-align: center;
        color: white;
        margin-bottom: 15px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* กำหนดสี Gradient พื้นหลังแยกตามประเภท */
    .card-total { background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 99%, #fecfef 100%); color: #333; }
    .card-car { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
    .card-moto { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: #333; }
    .card-heavy { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    
    /* สไตล์ตัวอักษรภายใน Card */
    .card-value { font-size: 2.8rem; font-weight: 700; margin: 5px 0; line-height: 1.2; }
    .card-label { font-size: 1.1rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# Helper Function สำหรับสร้าง HTML Card
def create_metric_card(label, value, card_class):
    return f"""
    <div class="dashboard-card {card_class}">
        <div class="card-label">{label}</div>
        <div class="card-value">{value}</div>
    </div>
    """

st.title("🚦 Traffic Analytics Dashboard")
st.markdown("---")

# โหลดโมเดล
@st.cache_resource
def load_tracker():
    return VehicleTracker('models/yolov8s.pt')

tracker = load_tracker()
video_path = "data/sample_video.mp4"

# 3. สร้าง UI Placeholders สำหรับ Card และแยก Column
col1, col2, col3, col4 = st.columns(4)
metric_total = col1.empty()
metric_car = col2.empty()
metric_motorcycle = col3.empty()
metric_truck_bus = col4.empty()

# แสดงผล Card เปล่าๆ เริ่มต้นก่อนกดรัน
metric_total.markdown(create_metric_card("Total Vehicles", "0", "card-total"), unsafe_allow_html=True)
metric_car.markdown(create_metric_card("🚗 Cars", "0", "card-car"), unsafe_allow_html=True)
metric_motorcycle.markdown(create_metric_card("🏍️ Motorcycles", "0", "card-moto"), unsafe_allow_html=True)
metric_truck_bus.markdown(create_metric_card("🚌 Heavy Vehicles", "0", "card-heavy"), unsafe_allow_html=True)

st.markdown("---")
main_col, chart_col = st.columns([6, 4])

with main_col:
    st.subheader("Live Video Feed")
    video_placeholder = st.empty()

with chart_col:
    st.subheader("Live Analytics")
    bar_chart_placeholder = st.empty()

# 4. ปุ่มเริ่มสตรีม
start_button = st.button("▶️ Start Video Stream")

if start_button:
    streamer = VideoStreamer(video_path)
    frame_count = 0
    
    for frame in streamer.generate_frames():
        frame_count += 1
        
        # ประมวลผล YOLO ทุกเฟรม
        processed_frame, total_count, class_counts = tracker.process_frame(frame)
        
        # กรองการอัปเดตหน้าจอ UI ทุกๆ 3 เฟรมเพื่อความสมูท
        if frame_count % 3 == 0:
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
            
            # อัปเดต HTML Cards แบบ Real-time
            metric_total.markdown(create_metric_card("Total Vehicles", total_count, "card-total"), unsafe_allow_html=True)
            metric_car.markdown(create_metric_card("🚗 Cars", class_counts["Car"], "card-car"), unsafe_allow_html=True)
            metric_motorcycle.markdown(create_metric_card("🏍️ Motorcycles", class_counts["Motorcycle"], "card-moto"), unsafe_allow_html=True)
            heavy_count = class_counts["Truck"] + class_counts["Bus"]
            metric_truck_bus.markdown(create_metric_card("🚌 Heavy Vehicles", heavy_count, "card-heavy"), unsafe_allow_html=True)
        
        # อัปเดตกราฟแท่ง ทุกๆ 30 เฟรม
        if frame_count % 30 == 0:
            df_bar = pd.DataFrame({
                "Class": ["Car", "Motorcycle", "Truck", "Bus"],
                "Count": [class_counts["Car"], class_counts["Motorcycle"], class_counts["Truck"], class_counts["Bus"]]
            })
            # ปรับแต่งกราฟ Plotly ให้ดูคลีนเข้ากับหน้า Dashboard
            fig_bar = px.bar(df_bar, x="Class", y="Count", color="Class", 
                             color_discrete_sequence=['#4facfe', '#38f9d7', '#764ba2', '#ff9a9e'])
            fig_bar.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=30, b=20),
                showlegend=False
            )
            bar_chart_placeholder.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{frame_count}")
            
    st.success("✅ วิดีโอประมวลผลเสร็จสิ้น!")