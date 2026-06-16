import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="ABSA Wisata Bandung Barat",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .main-header h1 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: #a8a5c8;
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }

    .metric-card h3 {
        color: #8b8fa3;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }

    .metric-card .value {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
    }

    .sentiment-positive {
        background: linear-gradient(135deg, #0d3b0d 0%, #1a5c1a 100%);
        border-left: 4px solid #2ecc71;
    }

    .sentiment-negative {
        background: linear-gradient(135deg, #3b0d0d 0%, #5c1a1a 100%);
        border-left: 4px solid #e74c3c;
    }

    .sentiment-neutral {
        background: linear-gradient(135deg, #0d2a3b 0%, #1a3e5c 100%);
        border-left: 4px solid #3498db;
    }

    .prediction-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }

    .section-header {
        color: #ffffff;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(255, 255, 255, 0.1);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Memuat model dan data...")
def load_app_data():
    """Memuat data dan model yang sudah di-train dari file pickle."""
    with open('models/app_data.pkl', 'rb') as f:
        app_data = f.read()
    data = pickle.loads(app_data)

    # Buat stemmer dan stopwords untuk prediksi real-time
    # (ringan, hanya membuat objek — tidak memproses seluruh dataset)
    stop_factory = StopWordRemoverFactory()
    stopwords_list = stop_factory.get_stop_words()

    stem_factory = StemmerFactory()
    stemmer = stem_factory.create_stemmer()

    data['stemmer'] = stemmer
    data['stopwords_list'] = stopwords_list

    return data


data = load_app_data()

df_clean = data['df_clean']
df_location = data['df_location']
final_models = data['final_models']
label_encoders = data['label_encoders']
tfidf_vectorizer = data['tfidf_vectorizer']
stemmer = data['stemmer']
stopwords_list = data['stopwords_list']
target_cols = data['target_cols']

st.markdown("""
<div class="main-header">
    <h1>🏔️ ABSA Dashboard - Wisata Alam Kabupaten Bandung Barat</h1>
    <p>Klasifikasi Sentimen Berbasis Aspek (Aksesibilitas, Fasilitas, Aktivitas)</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Total Ulasan</h3>
        <div class="value">{len(df_clean):,}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Lokasi Wisata</h3>
        <div class="value">{df_clean['location'].nunique()}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Jumlah Klaster</h3>
        <div class="value">{data['best_k']}</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    avg_f1 = np.mean([final_models[c]['f1'] for c in target_cols])
    st.markdown(f"""
    <div class="metric-card">
        <h3>Rata-rata F1</h3>
        <div class="value">{avg_f1:.2%}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Sentimen per Lokasi", "🗺️ Klaster Wisata", "🔮 Prediksi Real-Time"])

with tab1:
    st.markdown('<div class="section-header">Visualisasi Persentase Sentimen per Lokasi Wisata</div>', unsafe_allow_html=True)

    locations = sorted(df_location['location'].unique())
    selected_location = st.selectbox("Pilih Lokasi Wisata:", locations, index=0)

    loc_row = df_location[df_location['location'] == selected_location].iloc[0]

    st.markdown(f"**{selected_location}** | {int(loc_row['total_reviews'])} ulasan | Klaster {int(loc_row['cluster'])}")

    col_a, col_b, col_c = st.columns(3)

    aspect_names = {'accessibility': 'Aksesibilitas', 'facility': 'Fasilitas', 'activity': 'Aktivitas'}
    color_map = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#3498db'}

    for col_widget, col_name in zip([col_a, col_b, col_c], target_cols):
        with col_widget:
            st.markdown(f"<div style='text-align:center; font-weight:700; font-size:1.1rem; margin-bottom:-10px;'>{aspect_names[col_name]}</div>", unsafe_allow_html=True)
            pos = loc_row[f'{col_name}_positive_pct']
            neg = loc_row[f'{col_name}_negative_pct']
            neu = loc_row[f'{col_name}_neutral_pct']

            fig = go.Figure(data=[go.Pie(
                labels=['Positive', 'Negative', 'Neutral'],
                values=[pos, neg, neu],
                marker=dict(colors=['#2ecc71', '#e74c3c', '#3498db']),
                hole=0.55,
                textinfo='percent+label',
                textfont=dict(size=12, color='white'),
                hovertemplate='%{label}: %{value:.1f}%<extra></extra>'
            )])

            max_val = max(pos, neg, neu)
            if max_val == pos:
                maj_text, maj_color = 'Positif', '#2ecc71'
            elif max_val == neg:
                maj_text, maj_color = 'Negatif', '#e74c3c'
            else:
                maj_text, maj_color = 'Netral', '#3498db'

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=False,
                height=320,
                margin=dict(t=20, b=20, l=20, r=20),
                annotations=[dict(text=maj_text, x=0.5, y=0.5, font_size=22, font_weight='bold', font_color=maj_color, showarrow=False)]
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Perbandingan Semua Lokasi per Aspek</div>', unsafe_allow_html=True)

    selected_aspect = st.selectbox("Pilih Aspek:", target_cols, format_func=lambda x: aspect_names[x])

    df_sorted = df_location.sort_values(f'{selected_aspect}_positive_pct', ascending=True)

    fig_all = go.Figure()
    fig_all.add_trace(go.Bar(
        y=df_sorted['location'], x=df_sorted[f'{selected_aspect}_positive_pct'],
        name='Positive', orientation='h', marker_color='#2ecc71'
    ))
    fig_all.add_trace(go.Bar(
        y=df_sorted['location'], x=df_sorted[f'{selected_aspect}_negative_pct'],
        name='Negative', orientation='h', marker_color='#e74c3c'
    ))
    fig_all.add_trace(go.Bar(
        y=df_sorted['location'], x=df_sorted[f'{selected_aspect}_neutral_pct'],
        name='Neutral', orientation='h', marker_color='#3498db'
    ))

    fig_all.update_layout(
        barmode='stack',
        title=dict(text=f'Distribusi Sentimen {aspect_names[selected_aspect]} per Lokasi', font=dict(size=16, color='white')),
        xaxis_title='Persentase (%)',
        yaxis_title='',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        height=max(400, len(df_sorted) * 35),
        margin=dict(l=10, r=10)
    )
    fig_all.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig_all.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    st.plotly_chart(fig_all, use_container_width=True)


with tab2:
    st.markdown('<div class="section-header">Sebaran Klaster Wisata (K-Means Clustering)</div>', unsafe_allow_html=True)

    col_elbow, col_sil = st.columns(2)

    with col_elbow:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=data['K_range'], y=data['inertias'],
            mode='lines+markers+text',
            marker=dict(size=10, color='#3498db'),
            line=dict(width=3, color='#3498db'),
            text=[f'{v:.0f}' for v in data['inertias']],
            textposition='top center',
            textfont=dict(size=10, color='#3498db')
        ))
        fig_elbow.update_layout(
            title=dict(text='Elbow Method (Inertia/WCSS)', font=dict(size=15, color='white')),
            xaxis_title='Jumlah Klaster (K)',
            yaxis_title='Inertia',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=380,
            xaxis=dict(dtick=1, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col_sil:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(
            x=data['K_range'], y=data['sil_scores'],
            mode='lines+markers+text',
            marker=dict(size=10, color='#e74c3c'),
            line=dict(width=3, color='#e74c3c'),
            text=[f'{v:.3f}' for v in data['sil_scores']],
            textposition='top center',
            textfont=dict(size=10, color='#e74c3c')
        ))
        fig_sil.add_vline(x=data['best_k'], line_dash='dash', line_color='#2ecc71', line_width=2,
                          annotation_text=f"K optimal = {data['best_k']}", annotation_font_color='#2ecc71')
        fig_sil.update_layout(
            title=dict(text='Silhouette Score per K', font=dict(size=15, color='white')),
            xaxis_title='Jumlah Klaster (K)',
            yaxis_title='Silhouette Score',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=380,
            xaxis=dict(dtick=1, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_sil, use_container_width=True)

    st.markdown("---")

    cluster_colors = px.colors.qualitative.Set2
    df_location['cluster_label'] = df_location['cluster'].apply(lambda x: f'Klaster {x}')

    fig_scatter = px.scatter(
        df_location, x='pca_x', y='pca_y',
        color='cluster_label',
        hover_name='location',
        hover_data={'total_reviews': True, 'pca_x': False, 'pca_y': False, 'cluster_label': False},
        text='location',
        color_discrete_sequence=cluster_colors,
        title='Sebaran Klaster Destinasi Wisata (PCA 2D)'
    )
    fig_scatter.update_traces(
        textposition='top center',
        textfont=dict(size=10),
        marker=dict(size=14, line=dict(width=2, color='white'))
    )
    fig_scatter.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis_title='Komponen Utama 1 (PCA)',
        yaxis_title='Komponen Utama 2 (PCA)',
        height=550,
        legend_title='Klaster',
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Profil Sentimen per Klaster</div>', unsafe_allow_html=True)

    for cluster_id in sorted(df_location['cluster'].unique()):
        cluster_data = df_location[df_location['cluster'] == cluster_id]
        members = ', '.join(cluster_data['location'].tolist())

        with st.expander(f"📌 Klaster {cluster_id} ({len(cluster_data)} lokasi)", expanded=True):
            st.markdown(f"**Anggota:** {members}")

            fig_profile = make_subplots(rows=1, cols=3, subplot_titles=[aspect_names[c] for c in target_cols])

            for j, col_name in enumerate(target_cols):
                pos = cluster_data[f'{col_name}_positive_pct'].mean()
                neg = cluster_data[f'{col_name}_negative_pct'].mean()
                neu = cluster_data[f'{col_name}_neutral_pct'].mean()

                fig_profile.add_trace(go.Bar(
                    x=['Positif', 'Negatif', 'Netral'], y=[pos, neg, neu],
                    marker_color=['#2ecc71', '#e74c3c', '#3498db'],
                    text=[f'{v:.1f}%' for v in [pos, neg, neu]],
                    textposition='outside',
                    showlegend=False
                ), row=1, col=j+1)

            fig_profile.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=300,
                margin=dict(t=40, b=20)
            )
            fig_profile.update_yaxes(range=[0, 105], gridcolor='rgba(255,255,255,0.1)')
            fig_profile.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
            st.plotly_chart(fig_profile, use_container_width=True)


with tab3:
    st.markdown('<div class="section-header">🔮 Prediksi Sentimen Real-Time</div>', unsafe_allow_html=True)

    st.markdown("Ketik ulasan wisata dalam **Bahasa Indonesia**, dan model akan memprediksi sentimen untuk setiap aspek. Sistem juga akan **mencari lokasi wisata** yang ulasannya relevan dengan kata kunci Anda.")

    col_model_info = st.columns(3)
    for i, col_name in enumerate(target_cols):
        with col_model_info[i]:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{aspect_names[col_name]}</h3>
                <div class="value" style="font-size: 1rem;">{final_models[col_name]['name']}</div>
                <p style="color: #2ecc71; margin: 0;">F1: {final_models[col_name]['f1']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    user_input = st.text_area(
        "Tulis ulasan wisata di sini:",
        placeholder="Contoh: Jalannya rusak dan berlubang, pemandangannya indah. Toilet kotor tapi trekking seru.",
        height=120
    )

    if st.button("🔍 Prediksi Sentimen", use_container_width=True, type="primary"):
        if user_input.strip():
            with st.spinner("Memproses ulasan..."):
                text_processed = user_input.lower()
                text_processed = re.sub(r'http\S+|www\.\S+', '', text_processed)
                text_processed = re.sub(r'@\w+', '', text_processed)
                text_processed = re.sub(r'#\w+', '', text_processed)
                text_processed = re.sub(r'[^\w\s]', '', text_processed)
                text_processed = re.sub(r'\d+', '', text_processed)
                text_processed = re.sub(r'\s+', ' ', text_processed).strip()

                words = text_processed.split()
                filtered = [w for w in words if w not in stopwords_list]
                text_processed = ' '.join(filtered)

                text_processed = stemmer.stem(text_processed)

                X_input = tfidf_vectorizer.transform([text_processed])

                st.markdown("---")
                st.markdown("### Hasil Prediksi")

                result_cols = st.columns(3)
                sentiment_emoji = {'positive': '😊', 'negative': '😞', 'neutral': '😐'}
                sentiment_color = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#3498db'}
                sentiment_indo = {'positive': 'Positif', 'negative': 'Negatif', 'neutral': 'Netral'}

                for i, col_name in enumerate(target_cols):
                    with result_cols[i]:
                        model = final_models[col_name]['model']
                        pred_encoded = model.predict(X_input)[0]
                        pred_label = label_encoders[col_name].inverse_transform([pred_encoded])[0]

                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(X_input)[0]
                            confidence = proba[pred_encoded] * 100
                        else:
                            confidence = 0

                        emoji = sentiment_emoji.get(pred_label, '❓')
                        color = sentiment_color.get(pred_label, '#ffffff')
                        label_indo = sentiment_indo.get(pred_label, pred_label)

                        st.markdown(f"""
                        <div class="prediction-box" style="border-left: 4px solid {color}; text-align: center;">
                            <h4 style="color: #8b8fa3; margin: 0; font-size: 0.85rem; text-transform: uppercase;">{aspect_names[col_name]}</h4>
                            <div style="font-size: 3rem; margin: 0.3rem 0;">{emoji}</div>
                            <div style="color: {color}; font-size: 1.5rem; font-weight: 800;">{label_indo}</div>
                            <div style="color: #8b8fa3; font-size: 0.85rem; margin-top: 0.3rem;">Confidence: {confidence:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("### 📍 Lokasi Wisata yang Relevan")
                st.markdown("Lokasi berikut memiliki ulasan yang mengandung kata kunci serupa dengan input Anda:")

                input_keywords = [w for w in text_processed.split() if len(w) > 2]

                if input_keywords:
                    location_scores = {}
                    location_matched_kw = {}

                    for _, row in df_clean.iterrows():
                        review_words = set(str(row['text_clean']).split())
                        matched = [kw for kw in input_keywords if kw in review_words]
                        if matched:
                            loc = row['location']
                            if loc not in location_scores:
                                location_scores[loc] = 0
                                location_matched_kw[loc] = set()
                            location_scores[loc] += len(matched)
                            location_matched_kw[loc].update(matched)

                    if location_scores:
                        sorted_locations = sorted(location_scores.items(), key=lambda x: x[1], reverse=True)
                        top_locations = sorted_locations[:5]

                        max_score = top_locations[0][1]

                        for rank, (loc_name, score) in enumerate(top_locations):
                            loc_info = df_location[df_location['location'] == loc_name].iloc[0]
                            cluster_id = int(loc_info['cluster'])
                            total_reviews = int(loc_info['total_reviews'])
                            relevance_pct = score / max_score * 100
                            matched_kws = ', '.join(sorted(location_matched_kw[loc_name]))

                            cluster_members = df_location[df_location['cluster'] == cluster_id]['location'].tolist()
                            other_members = [m for m in cluster_members if m != loc_name]

                            with st.expander(f"{'🥇' if rank == 0 else '🥈' if rank == 1 else '🥉' if rank == 2 else '📌'} {loc_name} | Relevansi {relevance_pct:.0f}% ({score} kecocokan)", expanded=(rank == 0)):
                                info_cols = st.columns([2, 1])

                                with info_cols[0]:
                                    st.markdown(f"**Kata kunci cocok:** `{matched_kws}`")
                                    st.markdown(f"**Total ulasan di dataset:** {total_reviews}")
                                    if other_members:
                                        st.markdown(f"**Klaster {cluster_id}** | Lokasi serupa: {', '.join(other_members)}")
                                    else:
                                        st.markdown(f"**Klaster {cluster_id}**")

                                with info_cols[1]:
                                    st.markdown(f"""
                                    <div class="metric-card" style="padding: 1rem;">
                                        <h3>Relevansi</h3>
                                        <div class="value" style="color: {'#2ecc71' if relevance_pct > 70 else '#f39c12' if relevance_pct > 40 else '#e74c3c'};">{relevance_pct:.0f}%</div>
                                    </div>
                                    """, unsafe_allow_html=True)

                                st.markdown("**Profil Sentimen Lokasi:**")
                                profile_cols = st.columns(3)
                                for j, col_name in enumerate(target_cols):
                                    with profile_cols[j]:
                                        st.markdown(f"<div style='text-align:center; font-weight:700; font-size:1rem; margin-bottom:-10px;'>{aspect_names[col_name]}</div>", unsafe_allow_html=True)
                                        pos = loc_info[f'{col_name}_positive_pct']
                                        neg = loc_info[f'{col_name}_negative_pct']
                                        neu = loc_info[f'{col_name}_neutral_pct']

                                        fig_loc = go.Figure(data=[go.Pie(
                                            labels=['Positif', 'Negatif', 'Netral'],
                                            values=[pos, neg, neu],
                                            marker=dict(colors=['#2ecc71', '#e74c3c', '#3498db']),
                                            hole=0.55,
                                            textinfo='percent+label',
                                            textfont=dict(size=11, color='white'),
                                            hovertemplate='%{label}: %{value:.1f}%<extra></extra>'
                                        )])

                                        max_val = max(pos, neg, neu)
                                        if max_val == pos:
                                            maj_text, maj_color = 'Positif', '#2ecc71'
                                        elif max_val == neg:
                                            maj_text, maj_color = 'Negatif', '#e74c3c'
                                        else:
                                            maj_text, maj_color = 'Netral', '#3498db'

                                        fig_loc.update_layout(
                                            paper_bgcolor='rgba(0,0,0,0)',
                                            plot_bgcolor='rgba(0,0,0,0)',
                                            font=dict(color='white'),
                                            showlegend=False,
                                            height=260,
                                            margin=dict(t=10, b=10, l=10, r=10),
                                            annotations=[dict(text=maj_text, x=0.5, y=0.5, font_size=18, font_weight='bold', font_color=maj_color, showarrow=False)]
                                        )
                                        st.plotly_chart(fig_loc, use_container_width=True)

                    else:
                        st.info("Tidak ditemukan lokasi yang relevan dengan kata kunci tersebut.")
                else:
                    st.info("Kata kunci terlalu pendek untuk pencarian lokasi.")

                with st.expander("📝 Detail Preprocessing"):
                    st.markdown(f"**Teks asli:** {user_input}")
                    st.markdown(f"**Setelah preprocessing:** {text_processed}")
                    st.markdown(f"**Kata kunci pencarian:** {', '.join(input_keywords) if input_keywords else '-'}")

        else:
            st.warning("Silakan masukkan ulasan terlebih dahulu.")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #555; padding: 1rem;'>"
    "Tugas Besar Data Mining - Klasifikasi Sentimen Berbasis Aspek (ABSA) pada Ulasan Wisata Alam Kabupaten Bandung Barat"
    "</div>",
    unsafe_allow_html=True
)
