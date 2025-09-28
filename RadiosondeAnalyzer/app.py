import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from datetime import datetime, timedelta
import pickle
import os
import io
import calendar
from sklearn.preprocessing import LabelEncoder
from utils.data_processor import DataProcessor
from utils.model_trainer import ModelTrainer
from utils.predictor import Predictor
from utils.data_fetcher import DataFetcher

st.set_page_config(
    page_title="BMKG Tarakan Radiosonde analyzer",
    page_icon="🌧️",
    layout="wide"
)

def initialize_session_state():
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "home"
    if 'training_step' not in st.session_state:
        st.session_state.training_step = 1
    if 'rainfall_data' not in st.session_state:
        st.session_state.rainfall_data = None
    if 'radiosonde_data' not in st.session_state:
        st.session_state.radiosonde_data = None
    if 'rainfall_files_info' not in st.session_state:
        st.session_state.rainfall_files_info = []
    if 'radiosonde_files_info' not in st.session_state:
        st.session_state.radiosonde_files_info = []
    if 'merged_data' not in st.session_state:
        st.session_state.merged_data = None
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "Random Forest"
    if 'models' not in st.session_state:
        st.session_state.models = {}
    if 'uploaded_models' not in st.session_state:
        st.session_state.uploaded_models = {}
    if 'labels' not in st.session_state:
        st.session_state.labels = [
            {"min": 0, "max": 10, "label": "Berawan - hujan ringan"},
            {"min": 10, "max": float('inf'), "label": "Hujan sedang - lebat"}
        ]
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None
    if 'batch_prediction_results' not in st.session_state:
        st.session_state.batch_prediction_results = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'feature_columns' not in st.session_state:
        st.session_state.feature_columns = ['KI', 'LI', 'SI', 'TT', 'CAPE']
    if 'target_column' not in st.session_state:
        st.session_state.target_column = '6H'

def detect_header_row(df, required_features):
    """Detect if headers are in a data row instead of column names"""
    # Check first few rows for required features
    for row_idx in range(min(5, len(df))):
        row_values = [str(val).strip().upper() for val in df.iloc[row_idx].values if pd.notna(val)]
        found_features = []
        for feature in required_features:
            if feature.upper() in row_values:
                found_features.append(feature)
        
        if len(found_features) >= len(required_features) * 0.8:  # At least 80% of features found
            return row_idx
    
    return None

def fix_dataframe_structure(df, required_features):
    """Fix DataFrame structure when headers are in data rows"""
    header_row = detect_header_row(df, required_features)
    
    if header_row is not None:
        st.info(f"🔍 Mendeteksi header di baris {header_row}. Memperbaiki struktur data...")
        
        # Use the detected row as column names
        new_columns = []
        for col_val in df.iloc[header_row].values:
            if pd.notna(col_val):
                new_columns.append(str(col_val).strip())
            else:
                new_columns.append(f"Unnamed_{len(new_columns)}")
        
        # Create new dataframe with proper headers
        new_df = df.iloc[header_row + 1:].copy()  # Skip header row and above
        new_df.columns = new_columns[:len(new_df.columns)]
        new_df = new_df.reset_index(drop=True)
        
        return new_df
    
    return df

def clean_dataframe_for_display(df):
    """Clean DataFrame to avoid Arrow serialization issues"""
    if df is None or df.empty:
        return df
    
    cleaned_df = df.copy()
    
    # Clean column names - remove special characters and limit length
    new_columns = {}
    for col in cleaned_df.columns:
        # Convert to string and clean
        clean_col = str(col).strip()
        # Remove or replace problematic characters
        clean_col = clean_col.replace('\n', ' ').replace('\r', ' ')
        clean_col = clean_col.replace('  ', ' ')  # Replace double spaces
        # Limit column name length
        if len(clean_col) > 30:
            clean_col = clean_col[:27] + "..."
        new_columns[col] = clean_col
    
    cleaned_df = cleaned_df.rename(columns=new_columns)
    
    # Clean data types for each column
    for col in cleaned_df.columns:
        try:
            # Handle object columns that might have mixed types
            if cleaned_df[col].dtype == 'object':
                # Convert all values to string first to avoid mixed types
                cleaned_df[col] = cleaned_df[col].astype(str)
                
                # Try to convert to numeric if possible
                numeric_series = pd.to_numeric(cleaned_df[col], errors='coerce')
                if not numeric_series.isna().all():
                    # If more than 50% can be converted to numeric, use numeric
                    if numeric_series.notna().sum() > len(cleaned_df) * 0.5:
                        cleaned_df[col] = numeric_series.fillna(0)
                    else:
                        # Keep as string but clean
                        cleaned_df[col] = cleaned_df[col].replace('nan', '').replace('None', '')
            
            # Handle datetime columns
            elif 'datetime' in str(cleaned_df[col].dtype).lower():
                cleaned_df[col] = cleaned_df[col].astype(str)
            
            # Replace inf and -inf values
            if cleaned_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                cleaned_df[col] = cleaned_df[col].replace([np.inf, -np.inf], 0)
                cleaned_df[col] = cleaned_df[col].fillna(0)
        
        except Exception as e:
            # If there's any issue, convert to string and clean
            try:
                cleaned_df[col] = cleaned_df[col].astype(str)
                cleaned_df[col] = cleaned_df[col].replace('nan', '').replace('None', '')
            except:
                # Last resort - fill with empty string
                cleaned_df[col] = ""
    
    return cleaned_df

def safe_display_dataframe(df, use_container_width=True, key=None):
    """Safely display DataFrame with error handling"""
    try:
        if df is None or df.empty:
            st.write("No data to display")
            return
        
        cleaned_df = clean_dataframe_for_display(df)
        st.dataframe(cleaned_df, use_container_width=use_container_width, key=key)
    except Exception as e:
        st.error(f"Error displaying data: {str(e)}")
        # Fallback to basic table display
        try:
            st.write("**Data Preview (Basic Table):**")
            # Show only first few rows and columns to avoid issues
            display_df = df.head(10)
            if len(display_df.columns) > 10:
                display_df = display_df.iloc[:, :10]
            st.write(display_df.to_string())
        except:
            st.write("Unable to display data due to formatting issues")

def smart_column_detection(df, required_features):
    """Smart detection of required columns even if they're in different positions"""
    column_mapping = {}
    
    # First, try direct column name matching
    for feature in required_features:
        for col in df.columns:
            if str(col).strip().upper() == feature.upper():
                column_mapping[feature] = col
                break
    
    # If not all features found, search in data rows
    if len(column_mapping) < len(required_features):
        # Check if headers are in data rows
        fixed_df = fix_dataframe_structure(df, required_features)
        if not fixed_df.equals(df):
            # Structure was fixed, try again
            for feature in required_features:
                if feature not in column_mapping:
                    for col in fixed_df.columns:
                        if str(col).strip().upper() == feature.upper():
                            column_mapping[feature] = col
                            break
            return column_mapping, fixed_df
    
    return column_mapping, df

def safe_create_chart(chart_type, data=None, **kwargs):
    """Safely create plotly charts with error handling"""
    try:
        if chart_type == "bar":
            if data is not None:
                fig = px.bar(data, **kwargs)
            else:
                # Handle direct x, y parameters
                fig = px.bar(**kwargs)
        elif chart_type == "pie":
            if data is not None:
                fig = px.pie(data, **kwargs)
            else:
                # Handle direct values, names parameters
                fig = px.pie(**kwargs)
        elif chart_type == "histogram":
            if data is not None:
                fig = px.histogram(data, **kwargs)
            else:
                fig = px.histogram(**kwargs)
        elif chart_type == "box":
            if data is not None:
                fig = px.box(data, **kwargs)
            else:
                fig = px.box(**kwargs)
        else:
            return None
        
        return fig
    except Exception as e:
        st.error(f"Error creating {chart_type} chart: {str(e)}")
        return None

def show_home():
    st.title("🌧️ BMKG Tarakan Rainfall Prediction System")
    st.markdown("---")
    
    st.markdown("""
    ### Selamat Datang di Sistem Prediksi Curah Hujan BMKG Tarakan
    
    Sistem ini menggunakan teknologi machine learning untuk memprediksi curah hujan berdasarkan data radiosonde.
    Sistem terdiri dari tiga modul utama:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background-color: #f0fff0; padding: 20px; border-radius: 10px; margin: 10px 0;">
        <h3>🔮 Prediksi</h3>
        <p>Modul untuk melakukan prediksi curah hujan menggunakan model yang telah dilatih.</p>
        <ul>
        <li>Pengambilan data radiosonde real-time</li>
        <li>Upload file untuk prediksi batch</li>
        <li>Pemilihan model prediksi</li>
        <li>Analisis statistik data</li>
        <li>Visualisasi hasil prediksi</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔍 Mulai Prediksi", use_container_width=True):
            st.session_state.current_page = "prediction"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div style="background-color: #fff8dc; padding: 20px; border-radius: 10px; margin: 10px 0;">
        <h3>📤 Upload Model</h3>
        <p>Modul untuk mengupload dan menggunakan model eksternal untuk prediksi.</p>
        <ul>
        <li>Upload model dari device</li>
        <li>Validasi kompatibilitas model</li>
        <li>Prediksi dengan model eksternal</li>
        <li>Manajemen model tersimpan</li>
        <li>Kustomisasi nama kelas prediksi</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📤 Upload Model", use_container_width=True):
            st.session_state.current_page = "upload_model"
            st.rerun()
    
    with col3:
        st.markdown("""
        <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; margin: 10px 0;">
        <h3>🎯 Training Data</h3>
        <p>Modul untuk pelatihan model machine learning menggunakan data historis.</p>
        <ul>
        <li>Upload multiple dataset files</li>
        <li>Pembersihan data otomatis</li>
        <li>Konfigurasi labeling</li>
        <li>Pelatihan dengan berbagai algoritma</li>
        <li>Opsi SMOTE oversampling</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Mulai Training", use_container_width=True):
            st.session_state.current_page = "training"
            st.session_state.training_step = 1
            st.rerun()
    
    st.markdown("---")
    
    all_models = {**st.session_state.models, **st.session_state.uploaded_models}
    if len(all_models) > 0:
        st.subheader("📊 Model yang Tersedia")
        col1, col2, col3 = st.columns(3)
        
        for i, (model_name, model_data) in enumerate(all_models.items()):
            with [col1, col2, col3][i % 3]:
                if 'metrics' in model_data and 'accuracy' in model_data['metrics']:
                    accuracy = model_data['metrics']['accuracy']
                    model_type = "Training" if model_name in st.session_state.models else "Upload"
                    st.metric(f"{model_name} ({model_type})", f"{accuracy:.2%}", "Accuracy")
                else:
                    model_type = "Upload"
                    st.metric(f"{model_name} ({model_type})", "Siap Prediksi", "Status")
    else:
        st.info("Belum ada model yang tersedia. Silakan mulai dengan Training Data atau Upload Model untuk membuat/menambah model prediksi.")

def show_prediction_system():
    st.title("🔮 Sistem Prediksi Curah Hujan")
    
    if st.button("← Kembali ke Home"):
        st.session_state.current_page = "home"
        st.rerun()
    
    st.markdown("---")
    
    all_models = {**st.session_state.models, **st.session_state.uploaded_models}
    
    if len(all_models) == 0:
        st.warning("Tidak ada model yang tersedia untuk prediksi. Silakan latih model baru atau upload model yang sudah ada.")
        return
    
    # Pilihan metode prediksi
    st.subheader("🎯 Pilih Metode Prediksi")
    prediction_method = st.radio(
        "Metode Prediksi:",
        ["📅 Manual - Pilih Tanggal & Jam", "📄 Upload File - Prediksi Batch"],
        help="Pilih metode untuk melakukan prediksi curah hujan"
    )
    
    st.markdown("---")
    
    if prediction_method == "📅 Manual - Pilih Tanggal & Jam":
        show_manual_prediction(all_models)
    else:
        show_file_upload_prediction(all_models)

def show_manual_prediction(all_models):
    st.subheader("📅 Prediksi Manual - Pilih Tanggal dan Jam Pengamatan")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_date = st.date_input("Tanggal Mulai", value=datetime.now().date())
    
    with col2:
        end_date = st.date_input("Tanggal Akhir", value=datetime.now().date())
    
    with col3:
        obs_times = st.multiselect("Jam Pengamatan (UTC)", ["00", "12"], default=["00", "12"])
    
    if start_date > end_date:
        st.error("Tanggal mulai tidak boleh lebih besar dari tanggal akhir!")
        return
    
    if not obs_times:
        st.error("Pilih minimal satu jam pengamatan!")
        return
    
    st.subheader("🤖 Pilih Model untuk Prediksi")
    selected_models = st.multiselect(
        "Model yang akan digunakan",
        options=list(all_models.keys()),
        default=list(all_models.keys())
    )
    
    if not selected_models:
        st.warning("Pilih minimal satu model untuk prediksi!")
        return
    
    if st.button("🚀 Mulai Prediksi Manual", type="primary"):
        with st.spinner("Mengambil data radiosonde dan melakukan prediksi..."):
            try:
                data_fetcher = DataFetcher()
                predictor = Predictor()
                
                prediction_results = {}
                
                current_date = start_date
                while current_date <= end_date:
                    year = current_date.year
                    month = current_date.month
                    day = current_date.day
                    
                    for obs_time in obs_times:
                        try:
                            url = data_fetcher.generate_url(year, month, day, obs_time, day, obs_time)
                            raw_data = data_fetcher.fetch_rason_data(url)
                            
                            if raw_data:
                                rason_data, _, _, _ = data_fetcher.analyze_rason(raw_data, day, day)
                                
                                if rason_data:
                                    date_key = f"{current_date} {obs_time}:00"
                                    prediction_results[date_key] = {}
                                    
                                    for model_name in selected_models:
                                        model_data = all_models[model_name]
                                        model = model_data['model']
                                        label_rules = model_data.get('labels', st.session_state.labels)
                                        class_mapping = model_data.get('class_mapping', None)
                                        label_encoder = model_data.get('label_encoder', None)
                                        
                                        results = predictor.predict(rason_data, model, label_rules, class_mapping, label_encoder)
                                        if results:
                                            prediction_results[date_key][model_name] = results[0]
                        
                        except Exception as e:
                            st.warning(f"Gagal mengambil data untuk {current_date} {obs_time}:00 - {str(e)}")
                    
                    current_date += timedelta(days=1)
                
                if prediction_results:
                    st.session_state.prediction_results = prediction_results
                    st.success("Prediksi manual berhasil dilakukan!")
                    show_manual_prediction_results()
                else:
                    st.error("Tidak ada data yang berhasil diambil untuk rentang tanggal yang dipilih.")
            
            except Exception as e:
                st.error(f"Error dalam proses prediksi: {str(e)}")

def show_file_upload_prediction(all_models):
    st.subheader("📄 Upload File untuk Prediksi Batch")
    
    required_features = st.session_state.feature_columns
    
    st.write("**Fitur yang diperlukan untuk model:**")
    st.write(", ".join(required_features))
    
    uploaded_file = st.file_uploader(
        "Upload file CSV atau Excel dengan data untuk prediksi:",
        type=['csv', 'xlsx', 'xls'],
        help="Upload file CSV atau Excel yang berisi fitur KI, LI, SI, TT, CAPE untuk prediksi"
    )
    
    if uploaded_file is not None:
        try:
            # Load data berdasarkan tipe file
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(uploaded_file)
            else:
                st.error("❌ Format file tidak didukung")
                return
            
            st.subheader("📊 Preview Data")
            st.write(f"**Shape:** {data.shape}")
            safe_display_dataframe(data.head(), key="upload_data_preview")
            
            # Smart column detection
            column_mapping, processed_data = smart_column_detection(data, required_features)
            
            # Show detection results
            found_features = list(column_mapping.keys())
            missing_features = [f for f in required_features if f not in found_features]
            
            if found_features:
                st.success(f"✅ Fitur ditemukan: {', '.join(found_features)}")
            
            if missing_features:
                st.warning(f"⚠️ Fitur yang hilang: {', '.join(missing_features)}")
            
            # Show available columns for manual mapping
            st.subheader("🔗 Pemetaan Fitur")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Fitur yang Diperlukan:**")
                for feature in required_features:
                    status = "✅" if feature in found_features else "❌"
                    st.write(f"{status} {feature}")
            
            with col2:
                st.write("**Kolom yang Tersedia:**")
                for col in processed_data.columns:
                    st.write(f"• {col}")
            
            # Manual mapping untuk fitur yang hilang
            if missing_features:
                st.write("**Pemetaan Manual Fitur:**")
                for feature in missing_features:
                    mapped_col = st.selectbox(
                        f"Petakan '{feature}' ke:",
                        [""] + processed_data.columns.tolist(),
                        key=f"map_{feature}"
                    )
                    if mapped_col:
                        column_mapping[feature] = mapped_col
            
            # Pilih model untuk prediksi
            st.subheader("🤖 Pilih Model untuk Prediksi Batch")
            selected_models = st.multiselect(
                "Model yang akan digunakan untuk prediksi batch",
                options=list(all_models.keys()),
                default=list(all_models.keys())
            )
            
            if not selected_models:
                st.warning("Pilih minimal satu model untuk prediksi!")
                return
            
            # Tombol prediksi
            if len(column_mapping) == len(required_features) and selected_models:
                if st.button("🚀 Jalankan Prediksi Batch", type="primary"):
                    with st.spinner("Melakukan prediksi batch..."):
                        try:
                            # Siapkan data untuk prediksi
                            prediction_data = processed_data[[column_mapping[f] for f in required_features]]
                            prediction_data.columns = required_features
                            
                            # Buat prediksi untuk setiap model
                            predictor = Predictor()
                            batch_results = {}
                            
                            for model_name in selected_models:
                                model_data = all_models[model_name]
                                model = model_data['model']
                                label_rules = model_data.get('labels', st.session_state.labels)
                                class_mapping = model_data.get('class_mapping', None)
                                label_encoder = model_data.get('label_encoder', None)
                                
                                # Convert DataFrame ke list of dictionaries untuk predictor
                                data_list = prediction_data.to_dict('records')
                                
                                # Tambahkan index untuk identifikasi
                                for i, row in enumerate(data_list):
                                    row['Index'] = i
                                
                                results = predictor.predict(data_list, model, label_rules, class_mapping, label_encoder)
                                batch_results[model_name] = results
                            
                            # Buat results dataframe
                            results_df = processed_data.copy()
                            
                            # Tambahkan prediksi dari setiap model
                            for model_name, results in batch_results.items():
                                predictions = [r['prediction'] for r in results]
                                probabilities = [r['probability'] for r in results]
                                
                                results_df[f'Prediksi_{model_name}'] = predictions
                                results_df[f'Confidence_{model_name}'] = [f"{p:.2%}" for p in probabilities]
                            
                            st.session_state.batch_prediction_results = {
                                'results_df': results_df,
                                'batch_results': batch_results,
                                'selected_models': selected_models,
                                'prediction_data': prediction_data
                            }
                            
                            st.success("Prediksi batch berhasil dilakukan!")
                            show_batch_prediction_results()
                            
                        except Exception as e:
                            st.error(f"❌ Error selama prediksi: {str(e)}")
                            st.info("🔍 Silakan periksa format data dan pemetaan fitur.")
            else:
                if len(column_mapping) < len(required_features):
                    remaining = len(required_features) - len(column_mapping)
                    st.warning(f"⚠️ Silakan petakan {remaining} fitur lagi sebelum prediksi.")
                
        except Exception as e:
            st.error(f"❌ Error memuat file: {str(e)}")
            st.info("🔍 Pastikan file adalah CSV atau Excel yang valid.")

def show_manual_prediction_results():
    if st.session_state.prediction_results:
        st.subheader("📊 Hasil Prediksi Manual")
        
        results_data = []
        feature_data = []
        
        for date_time, models_results in st.session_state.prediction_results.items():
            for model_name, result in models_results.items():
                results_data.append({
                    'Waktu': date_time,
                    'Model': model_name,
                    'KI': result['KI'],
                    'LI': result['LI'],
                    'SI': result['SI'],
                    'TT': result['TT'],
                    'CAPE': result['CAPE'],
                    'Prediksi': result['prediction'],
                    'Confidence': f"{result['probability']:.2%}"
                })
                
                # Collect feature data for statistics - ensure numeric values
                feature_data.append({
                    'KI': pd.to_numeric(result['KI'], errors='coerce'),
                    'LI': pd.to_numeric(result['LI'], errors='coerce'),
                    'SI': pd.to_numeric(result['SI'], errors='coerce'),
                    'TT': pd.to_numeric(result['TT'], errors='coerce'),
                    'CAPE': pd.to_numeric(result['CAPE'], errors='coerce')
                })
        
        if results_data:
            results_df = pd.DataFrame(results_data)
            features_df = pd.DataFrame(feature_data)
            
            safe_display_dataframe(results_df, key="manual_results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Distribusi Prediksi")
                prediction_counts = results_df['Prediksi'].value_counts()
                fig_pie = safe_create_chart("pie", 
                    data=None,
                    values=prediction_counts.values, 
                    names=prediction_counts.index,
                    title="Distribusi Kategori Prediksi"
                )
                if fig_pie:
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.subheader("📊 Statistik Data")
                
                # Create statistics table
                stats_data = []
                for feature in ['KI', 'LI', 'SI', 'TT', 'CAPE']:
                    if feature in features_df.columns:
                        feature_values = features_df[feature].dropna()
                        if len(feature_values) > 0:
                            stats_data.append({
                                'Index': f"{feature}",
                                'Min': f"{feature_values.min():.2f}",
                                'Max': f"{feature_values.max():.2f}",
                                'Mean': f"{feature_values.mean():.2f}"
                            })
                        else:
                            stats_data.append({
                                'Index': f"{feature}",
                                'Min': "N/A",
                                'Max': "N/A",
                                'Mean': "N/A"
                            })
                
                if stats_data:
                    stats_df = pd.DataFrame(stats_data)
                    st.write("**Statistik Fitur yang Digunakan:**")
                    safe_display_dataframe(stats_df, key="manual_stats_table")
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Hasil Prediksi Manual (CSV)",
                data=csv,
                file_name=f"prediksi_manual_curah_hujan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def show_batch_prediction_results():
    if st.session_state.batch_prediction_results:
        batch_data = st.session_state.batch_prediction_results
        results_df = batch_data['results_df']
        batch_results = batch_data['batch_results']
        selected_models = batch_data['selected_models']
        
        st.subheader("📊 Hasil Prediksi Batch")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Prediksi", len(results_df))
        with col2:
            st.metric("Model Digunakan", len(selected_models))
        with col3:
            st.metric("Jumlah Data", len(results_df))
        
        # Tampilkan tabel hasil
        safe_display_dataframe(results_df, key="batch_results")
        
        # Visualisasi
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Distribusi Prediksi per Model")
            for model_name in selected_models:
                pred_col = f'Prediksi_{model_name}'
                if pred_col in results_df.columns:
                    pred_counts = results_df[pred_col].value_counts()
                    fig = safe_create_chart("pie",
                        data=None,
                        values=pred_counts.values,
                        names=pred_counts.index,
                        title=f"Distribusi Prediksi - {model_name}"
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Statistik Data yang Diproses")
            
            # Extract statistics from the results table (KI, LI, SI, TT, CAPE columns)
            feature_columns = ['KI', 'LI', 'SI', 'TT', 'CAPE']
            available_features = [col for col in feature_columns if col in results_df.columns]
            
            if available_features:
                stats_data = []
                for feature in feature_columns:
                    if feature in results_df.columns:
                        # Convert to numeric and handle any string values
                        feature_values = pd.to_numeric(results_df[feature], errors='coerce').dropna()
                        if len(feature_values) > 0:
                            stats_data.append({
                                'Index': f"**{feature}**",
                                'Min': f"{feature_values.min():.2f}",
                                'Max': f"{feature_values.max():.2f}",
                                'Mean': f"{feature_values.mean():.2f}"
                            })
                        else:
                            stats_data.append({
                                'Index': f"**{feature}**",
                                'Min': "N/A",
                                'Max': "N/A",
                                'Mean': "N/A"
                            })
                    else:
                        stats_data.append({
                            'Index': f"**{feature}**",
                            'Min': "N/A",
                            'Max': "N/A",
                            'Mean': "N/A"
                        })
                
                if stats_data:
                    stats_df = pd.DataFrame(stats_data)
                    st.write("**Statistik Fitur Input:**")
                    safe_display_dataframe(stats_df, key="batch_stats_table")
            else:
                st.write("Data fitur tidak tersedia untuk statistik")
        
        # Download hasil
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Hasil Prediksi Batch (CSV)",
            data=csv,
            file_name=f"prediksi_batch_curah_hujan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def show_upload_model_system():
    st.title("📤 Upload Model Eksternal")
    
    if st.button("← Kembali ke Home"):
        st.session_state.current_page = "home"
        st.rerun()
    
    st.markdown("---")
    
    st.info("⚠️ **Peringatan**: Model yang diupload harus memiliki target column yang sama dengan sistem ini (KI, LI, SI, TT, CAPE → Prediksi Curah Hujan)")
    
    st.subheader("📁 Upload Model (Format .pkl)")
    
    uploaded_file = st.file_uploader("Pilih file model (.pkl)", type=["pkl"])
    
    if uploaded_file is not None:
        model_name = st.text_input("Nama Model", value=uploaded_file.name.replace('.pkl', ''))
        
        if st.button("🚀 Upload dan Validasi Model", type="primary"):
            try:
                model_data = pickle.load(uploaded_file)
                
                # Check if it's a complete model package with label encoder
                if isinstance(model_data, dict) and 'model' in model_data:
                    model = model_data['model']
                    label_encoder = model_data.get('label_encoder', None)
                else:
                    model = model_data
                    label_encoder = None
                
                if hasattr(model, 'predict') and hasattr(model, 'predict_proba'):
                    try:
                        test_features = np.array([[35.0, -3.0, 0.5, 43.0, 1500.0]])
                        test_pred = model.predict(test_features)
                        test_proba = model.predict_proba(test_features)
                        
                        # Get unique classes from the model
                        model_classes = model.classes_
                        
                        # Convert prediction back to string if label encoder exists
                        if label_encoder is not None:
                            try:
                                display_pred = label_encoder.inverse_transform(test_pred)[0]
                            except:
                                display_pred = test_pred[0]
                        else:
                            display_pred = test_pred[0]
                        
                        st.session_state.uploaded_models[model_name] = {
                            'model': model,
                            'upload_date': datetime.now(),
                            'labels': st.session_state.labels,
                            'model_classes': model_classes.tolist(),
                            'class_mapping': {},  # Initialize empty class mapping
                            'label_encoder': label_encoder
                        }
                        
                        st.success(f"✅ Model '{model_name}' berhasil diupload dan siap digunakan!")
                        st.write(f"**Test Prediksi**: {display_pred}")
                        st.write(f"**Test Probabilitas**: {test_proba[0]}")
                        st.write(f"**Kelas Model**: {model_classes.tolist()}")
                        if label_encoder is not None:
                            st.write("**Label Encoder**: ✅ Tersedia")
                        
                        # Show class mapping configuration
                        st.markdown("---")
                        st.subheader("🏷️ Konfigurasi Nama Kelas")
                        st.info("Anda dapat mengubah nama kelas prediksi sesuai keinginan. Contoh: 0 → 'Cerah', 1 → 'Berawan', 2 → 'Hujan'")
                        
                        # Create class mapping interface
                        class_mapping = {}
                        for class_val in model_classes:
                            # If label encoder exists, show the original string labels
                            if label_encoder is not None:
                                try:
                                    original_label = label_encoder.inverse_transform([class_val])[0]
                                    default_name = str(original_label)
                                except:
                                    default_name = str(class_val)
                            else:
                                default_name = str(class_val)
                            
                            custom_name = st.text_input(
                                f"Nama untuk kelas '{class_val}':",
                                value=default_name,
                                key=f"class_name_{model_name}_{class_val}"
                            )
                            if custom_name and custom_name != str(class_val):
                                class_mapping[str(class_val)] = custom_name
                        
                        if st.button("💾 Simpan Konfigurasi Kelas", key=f"save_class_{model_name}"):
                            st.session_state.uploaded_models[model_name]['class_mapping'] = class_mapping
                            st.success("✅ Konfigurasi nama kelas berhasil disimpan!")
                            st.write("**Pemetaan Kelas:**")
                            for original, custom in class_mapping.items():
                                st.write(f"• {original} → {custom}")
                        
                    except Exception as e:
                        st.error(f"Model tidak kompatibel dengan format data sistem: {str(e)}")
                        st.error("Pastikan model dapat menerima 5 fitur input: KI, LI, SI, TT, CAPE")
                
                else:
                    st.error("Model tidak memiliki method predict dan predict_proba yang diperlukan!")
            
            except Exception as e:
                st.error(f"Error saat memuat model: {str(e)}")
    
    if st.session_state.uploaded_models:
        st.subheader("📋 Model yang Telah Diupload")
        
        for model_name, model_data in st.session_state.uploaded_models.items():
            with st.expander(f"📊 {model_name}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Upload:** {model_data['upload_date'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write("**Status:** ✅ Siap Prediksi")
                    st.write("**Type:** Model Eksternal")
                    st.write(f"**Kelas Model:** {model_data.get('model_classes', [])}")
                    if model_data.get('label_encoder') is not None:
                        st.write("**Label Encoder:** ✅ Tersedia")
                
                with col2:
                    st.write("**Pemetaan Kelas Saat Ini:**")
                    class_mapping = model_data.get('class_mapping', {})
                    if class_mapping:
                        for original, custom in class_mapping.items():
                            st.write(f"• {original} → {custom}")
                    else:
                        st.write("Menggunakan nama kelas asli")
                
                # Edit class mapping
                st.write("**Edit Nama Kelas:**")
                model_classes = model_data.get('model_classes', [])
                new_class_mapping = {}
                
                for class_val in model_classes:
                    current_name = class_mapping.get(str(class_val), str(class_val))
                    custom_name = st.text_input(
                        f"Nama untuk kelas '{class_val}':",
                        value=current_name,
                        key=f"edit_class_{model_name}_{class_val}"
                    )
                    if custom_name and custom_name != str(class_val):
                        new_class_mapping[str(class_val)] = custom_name
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Update Kelas", key=f"update_{model_name}"):
                        st.session_state.uploaded_models[model_name]['class_mapping'] = new_class_mapping
                        st.success("✅ Nama kelas berhasil diupdate!")
                        st.rerun()
                
                with col2:
                    if st.button("🗑️ Hapus Model", key=f"delete_{model_name}"):
                        del st.session_state.uploaded_models[model_name]
                        st.success(f"Model {model_name} berhasil dihapus!")
                        st.rerun()

def show_training_system():
    st.title("🎯 Training Data System")
    
    steps = [
        "📁 Upload Dataset", 
        "🔍 Data Preview & Cleaning", 
        "🏷️ Labeling Configuration", 
        "🤖 Model Training", 
        "📈 Evaluation & Model Saving"
    ]
    
    step_container = st.container()
    with step_container:
        cols = st.columns(len(steps))
        for i, step in enumerate(steps):
            with cols[i]:
                if i + 1 == st.session_state.training_step:
                    st.markdown(f"**{step}** ✅")
                elif i + 1 < st.session_state.training_step:
                    st.markdown(f"~~{step}~~ ✅")
                else:
                    st.markdown(f"{step}")
    
    st.markdown("---")
    
    if st.button("← Kembali ke Home"):
        st.session_state.current_page = "home"
        st.rerun()
    
    if st.session_state.training_step == 1:
        show_upload_step()
    elif st.session_state.training_step == 2:
        show_preview_step()
    elif st.session_state.training_step == 3:
        show_labeling_step()
    elif st.session_state.training_step == 4:
        show_training_step()
    elif st.session_state.training_step == 5:
        show_evaluation_step()

def show_upload_step():
    st.subheader("📁 Step 1: Upload Multiple Dataset Files")
    
    st.info("""
    **Format Data yang Diharapkan:**
    
    **Dataset Curah Hujan:**
    - Kolom: DATA, TIMESTAMP, 6H
    - Format tanggal: DD/MM/YYYY
    - Format waktu: HH:MM:SS.0 +0:00
    - Data curah hujan di kolom '6H'
    
    **Dataset Radiosonde:**
    - Kolom: TANGGAL, JAM, KI, LI, SI, TT, CAPE, BULAN
    - TANGGAL: 01, 02, 03, dst.
    - JAM: 00, 12 (UTC)
    - Tahun akan diambil dari nama file (contoh: dataset2022.csv, dataset2023.csv)
    
    **✨ Fitur Baru: Upload Multiple Files**
    - Anda dapat mengupload beberapa file sekaligus
    - File akan digabungkan secara otomatis
    - Mendukung data multi-tahun
    """)
    
    upload_col1, upload_col2 = st.columns(2)
    
    with upload_col1:
        st.write("**📊 Upload Multiple Data Curah Hujan**")
        rainfall_files = st.file_uploader(
            "Pilih file CSV atau Excel untuk data curah hujan", 
            type=["csv", "xlsx"], 
            key="rainfall_uploader",
            accept_multiple_files=True,
            help="Anda dapat memilih beberapa file sekaligus dengan Ctrl+Click atau Cmd+Click"
        )
        
        if rainfall_files:
            try:
                data_processor = DataProcessor()
                combined_rainfall, rainfall_info = data_processor.process_multiple_rainfall_files(rainfall_files)
                
                if combined_rainfall is not None:
                    st.session_state.rainfall_data = combined_rainfall
                    st.session_state.rainfall_files_info = rainfall_info
                    
                    st.success(f"✅ {len(rainfall_files)} file curah hujan berhasil dimuat!")
                    
                    # Show file information
                    st.write("**📋 Informasi File yang Diupload:**")
                    for info in rainfall_info:
                        st.write(f"• **{info['filename']}**: {info['rows']} baris, {info['columns']} kolom")
                    
                    st.write(f"**📊 Total Data Gabungan:** {len(combined_rainfall)} baris")
                    
                    st.write("**Preview Data Curah Hujan Gabungan:**")
                    safe_display_dataframe(combined_rainfall.head(), key="rainfall_preview")
                    
                    # Show column info
                    st.write("**Kolom yang ditemukan:**", list(combined_rainfall.columns))
                else:
                    st.error("❌ Gagal memproses file curah hujan")
                    
            except Exception as e:
                st.error(f"❌ Error memuat data curah hujan: {e}")
    
    with upload_col2:
        st.write("**🌡️ Upload Multiple Data Radiosonde**")
        radiosonde_files = st.file_uploader(
            "Pilih file CSV atau Excel untuk data radiosonde", 
            type=["csv", "xlsx"], 
            key="radiosonde_uploader",
            accept_multiple_files=True,
            help="Anda dapat memilih beberapa file sekaligus dengan Ctrl+Click atau Cmd+Click"
        )
        
        if radiosonde_files:
            try:
                data_processor = DataProcessor()
                combined_radiosonde, radiosonde_info = data_processor.process_multiple_radiosonde_files(radiosonde_files)
                
                if combined_radiosonde is not None:
                    st.session_state.radiosonde_data = combined_radiosonde
                    st.session_state.radiosonde_files_info = radiosonde_info
                    
                    st.success(f"✅ {len(radiosonde_files)} file radiosonde berhasil dimuat!")
                    
                    # Show file information
                    st.write("**📋 Informasi File yang Diupload:**")
                    for info in radiosonde_info:
                        st.write(f"• **{info['filename']}**: {info['rows']} baris, {info['columns']} kolom, Tahun: {info['year']}")
                    
                    st.write(f"**📊 Total Data Gabungan:** {len(combined_radiosonde)} baris")
                    
                    st.write("**Preview Data Radiosonde Gabungan:**")
                    safe_display_dataframe(combined_radiosonde.head(), key="radiosonde_preview")
                    
                    # Show column info
                    st.write("**Kolom yang ditemukan:**", list(combined_radiosonde.columns))
                    
                    # Show year distribution
                    if 'EXTRACTED_YEAR' in combined_radiosonde.columns:
                        year_counts = combined_radiosonde['EXTRACTED_YEAR'].value_counts().sort_index()
                        st.write("**📅 Distribusi Data per Tahun:**")
                        for year, count in year_counts.items():
                            st.write(f"• **{year}**: {count} record")
                else:
                    st.error("❌ Gagal memproses file radiosonde")
                    
            except Exception as e:
                st.error(f"❌ Error memuat data radiosonde: {e}")
    
    if st.session_state.rainfall_data is not None and st.session_state.radiosonde_data is not None:
        st.markdown("---")
        st.write("**🔗 Integrasi Multiple Dataset**")
        
        # Show summary of uploaded files
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Summary Data Curah Hujan:**")
            st.write(f"• Jumlah file: {len(st.session_state.rainfall_files_info)}")
            total_rainfall_rows = sum([info['rows'] for info in st.session_state.rainfall_files_info])
            st.write(f"• Total baris: {total_rainfall_rows}")
            
        with col2:
            st.write("**🌡️ Summary Data Radiosonde:**")
            st.write(f"• Jumlah file: {len(st.session_state.radiosonde_files_info)}")
            total_radiosonde_rows = sum([info['rows'] for info in st.session_state.radiosonde_files_info])
            st.write(f"• Total baris: {total_radiosonde_rows}")
            
            # Show year range
            years = [info['year'] for info in st.session_state.radiosonde_files_info]
            if years:
                st.write(f"• Rentang tahun: {min(years)} - {max(years)}")
        
        st.info("""
        **🔄 Proses Penggabungan Multiple Files:**
        1. Menggabungkan semua file curah hujan menjadi satu dataset
        2. Menggabungkan semua file radiosonde dengan informasi tahun dari nama file
        3. Mengambil data curah hujan pada jam 00:00 dan 12:00 saja
        4. Mencocokkan dengan data radiosonde berdasarkan tanggal, jam, dan tahun
        5. Hanya data yang cocok di kedua dataset yang akan disimpan
        """)
        
        if st.button("🚀 Gabungkan Multiple Dataset", type="primary"):
            with st.spinner("Menggabungkan multiple dataset..."):
                try:
                    data_processor = DataProcessor()
                    merged_data = data_processor.merge_datasets(
                        st.session_state.rainfall_data,
                        st.session_state.radiosonde_data
                    )
                    
                    st.session_state.merged_data = merged_data
                    st.session_state.processed_data = merged_data
                    st.success("✅ Multiple dataset berhasil digabungkan!")
                    
                    st.write("**📊 Preview Dataset Gabungan dari Multiple Files:**")
                    st.write(f"**Jumlah data:** {len(merged_data)} baris")
                    st.write(f"**Rentang tanggal:** {merged_data['DATETIME'].min()} sampai {merged_data['DATETIME'].max()}")
                    
                    safe_display_dataframe(merged_data.head(10), key="merged_preview")
                    
                    # Show statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Data", len(merged_data))
                    with col2:
                        obs_00 = len(merged_data[merged_data['DATETIME'].dt.hour == 0])
                        st.metric("Pengamatan 00:00", obs_00)
                    with col3:
                        obs_12 = len(merged_data[merged_data['DATETIME'].dt.hour == 12])
                        st.metric("Pengamatan 12:00", obs_12)
                    with col4:
                        unique_years = merged_data['DATETIME'].dt.year.nunique()
                        st.metric("Tahun Data", unique_years)
                    
                    # Show data distribution by source file if available
                    if 'SOURCE_FILE' in merged_data.columns:
                        st.write("**📁 Distribusi Data per File Sumber:**")
                        source_dist = merged_data['SOURCE_FILE'].value_counts()
                        
                        if len(source_dist) > 0:
                            # Create a bar chart with safe method
                            fig = safe_create_chart("bar", 
                                data=None,
                                x=source_dist.index,
                                y=source_dist.values,
                                title="Distribusi Data per File Sumber",
                                labels={'x': 'File Sumber', 'y': 'Jumlah Data'}
                            )
                            if fig:
                                fig.update_layout(xaxis_tickangle=45)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Show table
                            source_df = pd.DataFrame({
                                'File Sumber': source_dist.index,
                                'Jumlah Data': source_dist.values,
                                'Persentase': (source_dist.values / len(merged_data) * 100).round(2)
                            })
                            safe_display_dataframe(source_df, key="source_distribution")
                        else:
                            st.write("Tidak ada informasi file sumber yang tersedia")
                    
                except Exception as e:
                    st.error(f"❌ Error menggabungkan multiple dataset: {e}")
                    st.write("**Debug Info:**")
                    st.write("Kolom data curah hujan:", list(st.session_state.rainfall_data.columns))
                    st.write("Kolom data radiosonde:", list(st.session_state.radiosonde_data.columns))
        
        if st.session_state.merged_data is not None:
            st.markdown("---")
            if st.button("➡️ Lanjut ke Step 2: Data Preview & Cleaning", type="primary"):
                st.session_state.training_step = 2
                st.rerun()

def show_preview_step():
    st.subheader("🔍 Step 2: Data Preview & Cleaning")
    
    if st.session_state.processed_data is not None:
        st.write("**📊 Data dari Multiple Files:**")
        st.write(f"**Jumlah data:** {len(st.session_state.processed_data)} baris")
        
        # Show data quality info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            missing_rainfall = st.session_state.processed_data['6H'].isna().sum()
            st.metric("Data Curah Hujan Kosong", missing_rainfall)
        
        with col2:
            feature_cols = ['KI', 'LI', 'SI', 'TT', 'CAPE']
            missing_features = st.session_state.processed_data[feature_cols].isna().sum().sum()
            st.metric("Data Radiosonde Kosong", missing_features)
        
        with col3:
            total_complete = len(st.session_state.processed_data.dropna())
            st.metric("Data Lengkap", total_complete)
        
        with col4:
            if 'SOURCE_FILE' in st.session_state.processed_data.columns:
                unique_sources = st.session_state.processed_data['SOURCE_FILE'].nunique()
                st.metric("File Sumber", unique_sources)
        
        safe_display_dataframe(st.session_state.processed_data, key="processed_data_preview")
        
        # Show data distribution
        st.write("**📈 Distribusi Data Curah Hujan:**")
        rainfall_stats = st.session_state.processed_data['6H'].describe()
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Statistik Curah Hujan:**")
            st.write(rainfall_stats)
        
        with col2:
            fig = safe_create_chart("histogram", 
                data=st.session_state.processed_data, 
                x='6H',
                title="Distribusi Curah Hujan (6H)",
                nbins=30
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Show temporal distribution if datetime available
        if 'DATETIME' in st.session_state.processed_data.columns:
            st.write("**📅 Distribusi Temporal Data:**")
            
            # Year distribution
            year_dist = st.session_state.processed_data['DATETIME'].dt.year.value_counts().sort_index()
            
            col1, col2 = st.columns(2)
            with col1:
                fig_year = safe_create_chart("bar",
                    data=None,
                    x=year_dist.index,
                    y=year_dist.values,
                    title="Distribusi Data per Tahun",
                    labels={'x': 'Tahun', 'y': 'Jumlah Data'}
                )
                if fig_year:
                    st.plotly_chart(fig_year, use_container_width=True)
            
            with col2:
                # Monthly distribution
                month_dist = st.session_state.processed_data['DATETIME'].dt.month.value_counts().sort_index()
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                fig_month = safe_create_chart("bar",
                    data=None,
                    x=[month_names[i-1] for i in month_dist.index],
                    y=month_dist.values,
                    title="Distribusi Data per Bulan",
                    labels={'x': 'Bulan', 'y': 'Jumlah Data'}
                )
                if fig_month:
                    st.plotly_chart(fig_month, use_container_width=True)
        
        data_processor = DataProcessor()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Bersihkan Data", type="primary"):
                try:
                    with st.spinner("Membersihkan data dari multiple files..."):
                        cleaned_data = data_processor.clean_data(st.session_state.processed_data)
                        st.session_state.processed_data = cleaned_data
                        st.success("✅ Data dari multiple files berhasil dibersihkan!")
                        st.info("Data dengan nilai kosong telah dihapus.")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Error membersihkan data: {e}")
        
        with col2:
            if st.button("← Kembali ke Step 1"):
                st.session_state.training_step = 1
                st.rerun()
        
        st.markdown("---")
        if st.button("➡️ Lanjut ke Step 3: Labeling Configuration", type="primary"):
            st.session_state.training_step = 3
            st.rerun()
    else:
        st.warning("⚠️ Tidak ada data yang tersedia. Silakan kembali ke Step 1 untuk upload multiple dataset.")
        if st.button("← Kembali ke Step 1"):
            st.session_state.training_step = 1
            st.rerun()

def show_labeling_step():
    st.subheader("🏷️ Step 3: Labeling Configuration")
    
    if st.session_state.processed_data is not None and len(st.session_state.processed_data) > 0:
        st.write("**🎯 Definisikan Kategori Curah Hujan untuk Multiple Dataset:**")
        st.info("Tentukan rentang nilai curah hujan untuk setiap kategori prediksi berdasarkan data gabungan dari multiple files.")
        
        # Show current rainfall distribution
        rainfall_data = st.session_state.processed_data['6H']
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Statistik Curah Hujan dari Multiple Files:**")
            st.write(f"• Minimum: {rainfall_data.min():.2f} mm")
            st.write(f"• Maksimum: {rainfall_data.max():.2f} mm")
            st.write(f"• Rata-rata: {rainfall_data.mean():.2f} mm")
            st.write(f"• Median: {rainfall_data.median():.2f} mm")
            st.write(f"• Total data: {len(rainfall_data)} record")
        
        with col2:
            fig = safe_create_chart("box", 
                data=None,
                y=rainfall_data, 
                title="Distribusi Curah Hujan (Multiple Files)"
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.write("**⚙️ Konfigurasi Label:**")
        
        labels_to_display = st.session_state.labels.copy()
        
        for i, label_rule in enumerate(labels_to_display):
            st.write(f"**Label {i+1}:**")
            col1, col2, col3, col4 = st.columns([1, 1, 2, 1])
            
            with col1:
                min_val = st.number_input(f"Nilai Min", value=float(label_rule["min"]), key=f"min_{i}")
            
            with col2:
                if label_rule["max"] == float('inf'):
                    max_val = st.number_input(f"Nilai Max", value=1000.0, key=f"max_{i}")
                    if max_val >= 1000:
                        max_val = float('inf')
                else:
                    max_val = st.number_input(f"Nilai Max", value=float(label_rule["max"]), key=f"max_{i}")
            
            with col3:
                label_name = st.text_input(f"Nama Label", value=label_rule["label"], key=f"label_{i}")
            
            with col4:
                if st.button("🗑️ Hapus", key=f"delete_{i}") and len(st.session_state.labels) > 1:
                    st.session_state.labels.pop(i)
                    st.rerun()
            
            st.session_state.labels[i] = {"min": min_val, "max": max_val, "label": label_name}
            
            # Show how many samples fall into this category
            if max_val == float('inf'):
                count = len(rainfall_data[(rainfall_data >= min_val)])
            else:
                count = len(rainfall_data[(rainfall_data >= min_val) & (rainfall_data < max_val)])
            
            percentage = (count / len(rainfall_data)) * 100
            st.write(f"   📈 Jumlah data dalam kategori ini: **{count}** sampel ({percentage:.1f}%)")
            st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Tambah Label Baru"):
                if len(st.session_state.labels) > 0:
                    last_max = st.session_state.labels[-1]["max"]
                    if last_max == float('inf'):
                        last_max = 50.0
                    st.session_state.labels.append({"min": last_max, "max": last_max + 10, "label": "Kategori Baru"})
                else:
                    st.session_state.labels.append({"min": 0, "max": 10, "label": "Kategori Baru"})
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset ke Default"):
                st.session_state.labels = [
                    {"min": 0, "max": 10, "label": "Berawan - hujan ringan"},
                    {"min": 10, "max": float('inf'), "label": "Hujan sedang - lebat"}
                ]
                st.rerun()
        
        st.markdown("---")
        st.write("**🔍 Preview Labeling dan Analisis Data Multiple Files:**")
        
        try:
            data_processor = DataProcessor()
            labeled_data = data_processor.apply_labels(st.session_state.processed_data, st.session_state.target_column, st.session_state.labels)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**📋 Data dengan Label (Preview 10 baris pertama):**")
                safe_display_dataframe(labeled_data, key="labeled_data_preview")
            
            with col2:
                st.write("**📊 Statistik Distribusi Label:**")
                label_counts = labeled_data['label'].value_counts()
                st.write(label_counts)
                
                fig_pie = safe_create_chart("pie",
                    data=None,
                    values=label_counts.values, 
                    names=label_counts.index,
                    title="Distribusi Label (Multiple Files)"
                )
                if fig_pie:
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            # Check for imbalanced data
            min_count = label_counts.min()
            max_count = label_counts.max()
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            if imbalance_ratio > 3:
                st.warning(f"⚠️ **Data tidak seimbang terdeteksi!** Rasio: {imbalance_ratio:.1f}:1")
                st.info("💡 Pertimbangkan untuk menggunakan SMOTE oversampling pada step training untuk mengatasi ketidakseimbangan data.")
            else:
                st.success("✅ Distribusi data cukup seimbang.")
        
        except Exception as e:
            st.error(f"❌ Error dalam preview labeling: {e}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Kembali ke Step 2"):
                st.session_state.training_step = 2
                st.rerun()
        
        with col2:
            if st.button("➡️ Lanjut ke Step 4: Model Training", type="primary"):
                st.session_state.training_step = 4
                st.rerun()
    else:
        st.warning("⚠️ Tidak ada data yang tersedia. Silakan kembali ke step sebelumnya.")

def show_training_step():
    st.subheader("🤖 Step 4: Model Training dengan Multiple Dataset")
    
    if st.session_state.processed_data is not None:
        st.write("**🎯 Pilih Algoritma Machine Learning:**")
        
        algorithms = ["Random Forest", "SVM", "Decision Tree", "K-Nearest Neighbors", "Gradient Boosting"]
        selected_algorithm = st.selectbox("Algoritma", algorithms)
        
        st.write("**⚙️ Konfigurasi Training:**")
        
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Ukuran Data Test (%)", 10, 50, 30) / 100
            random_state = st.number_input("Random State", value=42, min_value=1)
            
            # SMOTE Configuration
            st.write("**⚖️ Opsi SMOTE Oversampling:**")
            use_smote = st.checkbox(
                "Gunakan SMOTE untuk mengatasi ketidakseimbangan data",
                help="SMOTE (Synthetic Minority Oversampling Technique) akan membuat sampel sintetis untuk kelas minoritas"
            )
            
            if use_smote:
                k_neighbors = st.number_input(
                    "K-Neighbors untuk SMOTE", 
                    value=5, 
                    min_value=1, 
                    max_value=10,
                    help="Jumlah tetangga terdekat yang digunakan untuk membuat sampel sintetis"
                )
                smote_params = {"k_neighbors": k_neighbors}
            else:
                smote_params = None
        
        with col2:
            if selected_algorithm == "Random Forest":
                n_estimators = st.number_input("N Estimators", value=100, min_value=10)
                max_depth = st.number_input("Max Depth", value=10, min_value=1)
                params = {"n_estimators": n_estimators, "max_depth": max_depth, "random_state": random_state}
            elif selected_algorithm == "SVM":
                C = st.number_input("C", value=1.0, min_value=0.1)
                kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])
                params = {"C": C, "kernel": kernel, "probability": True, "random_state": random_state}
            elif selected_algorithm == "Decision Tree":
                max_depth = st.number_input("Max Depth", value=10, min_value=1)
                min_samples_split = st.number_input("Min Samples Split", value=2, min_value=2)
                params = {"max_depth": max_depth, "min_samples_split": min_samples_split, "random_state": random_state}
            elif selected_algorithm == "K-Nearest Neighbors":
                n_neighbors = st.number_input("N Neighbors", value=5, min_value=1)
                weights = st.selectbox("Weights", ["uniform", "distance"])
                params = {"n_neighbors": n_neighbors, "weights": weights}
            elif selected_algorithm == "Gradient Boosting":
                n_estimators = st.number_input("N Estimators", value=100, min_value=10)
                learning_rate = st.number_input("Learning Rate", value=0.1, min_value=0.01, max_value=1.0)
                params = {"n_estimators": n_estimators, "learning_rate": learning_rate, "random_state": random_state}
        
        # Show current data distribution before training
        if use_smote:
            st.write("**📊 Analisis Distribusi Data dari Multiple Files:**")
            data_processor = DataProcessor()
            labeled_data = data_processor.apply_labels(
                st.session_state.processed_data, 
                st.session_state.target_column, 
                st.session_state.labels
            )
            
            if not labeled_data.empty:
                current_distribution = labeled_data['label'].value_counts()
                st.write("**Distribusi saat ini:**")
                for label, count in current_distribution.items():
                    st.write(f"• {label}: {count} sampel")
                
                st.info("💡 SMOTE akan menyeimbangkan distribusi kelas dengan membuat sampel sintetis untuk kelas minoritas.")
        
        st.markdown("---")
        st.write("**🔧 Hyperparameter Tuning:**")

        use_hyperparameter_tuning = st.checkbox(
            "Gunakan Hyperparameter Tuning",
            help="Otomatis mencari kombinasi parameter terbaik untuk meningkatkan akurasi model"
        )

        if use_hyperparameter_tuning:
            col1, col2 = st.columns(2)
            
            with col1:
                tuning_method = st.selectbox(
                    "Metode Tuning",
                    ["GridSearchCV", "RandomizedSearchCV"],
                    help="GridSearch: mencoba semua kombinasi (lebih akurat tapi lambat), RandomSearch: sampling acak (lebih cepat)"
                )
                
                cv_folds = st.number_input(
                    "Cross-Validation Folds",
                    value=5,
                    min_value=3,
                    max_value=10,
                    help="Jumlah fold untuk cross-validation"
                )
                
                scoring_metric = st.selectbox(
                    "Scoring Metric",
                    ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"],
                    help="Metrik untuk evaluasi parameter terbaik"
                )
            
            with col2:
                if tuning_method == "RandomizedSearchCV":
                    n_iter = st.number_input(
                        "Jumlah Iterasi Random Search",
                        value=50,
                        min_value=10,
                        max_value=200,
                        help="Jumlah kombinasi parameter yang akan dicoba secara acak"
                    )
                
                # Parameter ranges berdasarkan algoritma yang dipilih
                st.write("**Parameter Ranges untuk Tuning:**")
                
                if selected_algorithm == "Random Forest":
                    tune_n_estimators = st.checkbox("Tune N Estimators", value=True)
                    if tune_n_estimators:
                        n_est_range = st.slider("N Estimators Range", 50, 500, (100, 300), step=50)
                    
                    tune_max_depth = st.checkbox("Tune Max Depth", value=True)
                    if tune_max_depth:
                        depth_range = st.slider("Max Depth Range", 3, 30, (5, 20))
                    
                    tune_min_samples_split = st.checkbox("Tune Min Samples Split", value=True)
                    if tune_min_samples_split:
                        split_range = st.slider("Min Samples Split Range", 2, 20, (2, 10))
                
                elif selected_algorithm == "SVM":
                    tune_c = st.checkbox("Tune C Parameter", value=True)
                    if tune_c:
                        c_range = st.select_slider("C Range", options=[0.01, 0.1, 1, 10, 100], value=(0.1, 10))
                    
                    tune_gamma = st.checkbox("Tune Gamma", value=True)
                    if tune_gamma:
                        gamma_options = st.multiselect("Gamma Options", ['scale', 'auto', 0.001, 0.01, 0.1, 1], default=['scale', 0.01, 0.1])
                
                elif selected_algorithm == "Decision Tree":
                    tune_max_depth = st.checkbox("Tune Max Depth", value=True)
                    if tune_max_depth:
                        depth_range = st.slider("Max Depth Range", 3, 30, (5, 20))
                    
                    tune_min_samples_split = st.checkbox("Tune Min Samples Split", value=True)
                    if tune_min_samples_split:
                        split_range = st.slider("Min Samples Split Range", 2, 20, (2, 10))
                    
                    tune_min_samples_leaf = st.checkbox("Tune Min Samples Leaf", value=True)
                    if tune_min_samples_leaf:
                        leaf_range = st.slider("Min Samples Leaf Range", 1, 10, (1, 5))
                
                elif selected_algorithm == "K-Nearest Neighbors":
                    tune_n_neighbors = st.checkbox("Tune N Neighbors", value=True)
                    if tune_n_neighbors:
                        neighbors_range = st.slider("N Neighbors Range", 3, 20, (3, 15))
                    
                    tune_weights = st.checkbox("Tune Weights", value=True)
                    if tune_weights:
                        weights_options = st.multiselect("Weights Options", ['uniform', 'distance'], default=['uniform', 'distance'])
                
                elif selected_algorithm == "Gradient Boosting":
                    tune_n_estimators = st.checkbox("Tune N Estimators", value=True)
                    if tune_n_estimators:
                        n_est_range = st.slider("N Estimators Range", 50, 300, (100, 200), step=50)
                    
                    tune_learning_rate = st.checkbox("Tune Learning Rate", value=True)
                    if tune_learning_rate:
                        lr_options = st.multiselect("Learning Rate Options", [0.01, 0.05, 0.1, 0.2, 0.3], default=[0.05, 0.1, 0.2])
                    
                    tune_max_depth = st.checkbox("Tune Max Depth", value=True)
                    if tune_max_depth:
                        depth_range = st.slider("Max Depth Range", 3, 10, (3, 8))

        # Estimasi waktu tuning
        if use_hyperparameter_tuning:
            st.info("⏱️ **Estimasi Waktu**: Hyperparameter tuning dapat memakan waktu 5-30 menit tergantung pada kompleksitas parameter dan ukuran data.")

        if st.button("🚀 Mulai Training dengan Multiple Dataset", type="primary"):
            with st.spinner("Melatih model dengan data dari multiple files..."):
                try:
                    model_trainer = ModelTrainer()
                    
                    # Prepare hyperparameter tuning parameters
                    tuning_params = None
                    if use_hyperparameter_tuning:
                        tuning_params = {
                            'method': tuning_method,
                            'cv_folds': cv_folds,
                            'scoring': scoring_metric,
                            'algorithm': selected_algorithm
                        }
                        
                        if tuning_method == "RandomizedSearchCV":
                            tuning_params['n_iter'] = n_iter
                        
                        # Build parameter grid based on selected algorithm
                        param_grid = {}
                        
                        if selected_algorithm == "Random Forest":
                            if tune_n_estimators:
                                param_grid['n_estimators'] = list(range(n_est_range[0], n_est_range[1] + 1, 50))
                            if tune_max_depth:
                                param_grid['max_depth'] = list(range(depth_range[0], depth_range[1] + 1))
                            if tune_min_samples_split:
                                param_grid['min_samples_split'] = list(range(split_range[0], split_range[1] + 1))
                        
                        elif selected_algorithm == "SVM":
                            if tune_c:
                                param_grid['C'] = [c for c in [0.01, 0.1, 1, 10, 100] if c_range[0] <= c <= c_range[1]]
                            if tune_gamma:
                                param_grid['gamma'] = gamma_options
                        
                        elif selected_algorithm == "Decision Tree":
                            if tune_max_depth:
                                param_grid['max_depth'] = list(range(depth_range[0], depth_range[1] + 1))
                            if tune_min_samples_split:
                                param_grid['min_samples_split'] = list(range(split_range[0], split_range[1] + 1))
                            if tune_min_samples_leaf:
                                param_grid['min_samples_leaf'] = list(range(leaf_range[0], leaf_range[1] + 1))
                        
                        elif selected_algorithm == "K-Nearest Neighbors":
                            if tune_n_neighbors:
                                param_grid['n_neighbors'] = list(range(neighbors_range[0], neighbors_range[1] + 1))
                            if tune_weights:
                                param_grid['weights'] = weights_options
                        
                        elif selected_algorithm == "Gradient Boosting":
                            if tune_n_estimators:
                                param_grid['n_estimators'] = list(range(n_est_range[0], n_est_range[1] + 1, 50))
                            if tune_learning_rate:
                                param_grid['learning_rate'] = lr_options
                            if tune_max_depth:
                                param_grid['max_depth'] = list(range(depth_range[0], depth_range[1] + 1))
                        
                        tuning_params['param_grid'] = param_grid
                    
                    # Call training with hyperparameter tuning
                    if use_hyperparameter_tuning:
                        result = model_trainer.train_model_with_hyperparameter_tuning(
                            st.session_state.processed_data,
                            st.session_state.feature_columns,
                            st.session_state.target_column,
                            st.session_state.labels,
                            selected_algorithm,
                            tuning_params,
                            test_size,
                            random_state,
                            use_smote=use_smote,
                            smote_params=smote_params
                        )
                        
                        X_train, X_test, y_train, y_test, model, y_pred, accuracy, report, conf_matrix, label_encoder, original_distribution, resampled_distribution, best_params, cv_results = result
                        
                        st.session_state.current_model = {
                            'model': model,
                            'algorithm': selected_algorithm,
                            'accuracy': accuracy,
                            'report': report,
                            'confusion_matrix': conf_matrix,
                            'labels': st.session_state.labels,
                            'train_data': (X_train, y_train),
                            'test_data': (X_test, y_test),
                            'predictions': y_pred,
                            'classes': model.classes_,
                            'use_smote': use_smote,
                            'smote_params': smote_params,
                            'label_encoder': label_encoder,
                            'original_distribution': original_distribution,
                            'resampled_distribution': resampled_distribution,
                            'total_data': len(st.session_state.processed_data),
                            'source_files': {
                                'rainfall_files': len(st.session_state.rainfall_files_info),
                                'radiosonde_files': len(st.session_state.radiosonde_files_info)
                            },
                            'hyperparameter_tuning': {
                                'used': True,
                                'method': tuning_method,
                                'best_params': best_params,
                                'cv_results': cv_results,
                                'scoring': scoring_metric
                            }
                        }
                        
                        st.success(f"✅ Model berhasil dilatih dengan hyperparameter tuning!")
                        st.success(f"🎯 Akurasi terbaik: {accuracy:.2%}")
                        st.info("📊 **Parameter Terbaik:**")
                        for param, value in best_params.items():
                            st.write(f"• **{param}**: {value}")
                        
                    else:
                        # Original training without hyperparameter tuning
                        X_train, X_test, y_train, y_test, model, y_pred, accuracy, report, conf_matrix, label_encoder, original_distribution, resampled_distribution = model_trainer.train_model_with_params(
                            st.session_state.processed_data,
                            st.session_state.feature_columns,
                            st.session_state.target_column,
                            st.session_state.labels,
                            selected_algorithm,
                            params,
                            test_size,
                            random_state,
                            use_smote=use_smote,
                            smote_params=smote_params
                        )
                        
                        st.session_state.current_model = {
                            'model': model,
                            'algorithm': selected_algorithm,
                            'accuracy': accuracy,
                            'report': report,
                            'confusion_matrix': conf_matrix,
                            'labels': st.session_state.labels,
                            'train_data': (X_train, y_train),
                            'test_data': (X_test, y_test),
                            'predictions': y_pred,
                            'classes': model.classes_,
                            'use_smote': use_smote,
                            'smote_params': smote_params,
                            'label_encoder': label_encoder,
                            'original_distribution': original_distribution,
                            'resampled_distribution': resampled_distribution,
                            'total_data': len(st.session_state.processed_data),
                            'source_files': {
                                'rainfall_files': len(st.session_state.rainfall_files_info),
                                'radiosonde_files': len(st.session_state.radiosonde_files_info)
                            },
                            'hyperparameter_tuning': {
                                'used': False
                            }
                        }
                        
                        st.success(f"✅ Model berhasil dilatih dengan akurasi: {accuracy:.2%}")
                    
                    st.info(f"📁 Data training berasal dari {len(st.session_state.rainfall_files_info)} file curah hujan dan {len(st.session_state.radiosonde_files_info)} file radiosonde")
                    if use_smote:
                        st.info("✅ SMOTE oversampling telah diterapkan pada data training")
                    st.session_state.training_step = 5
                    st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Error dalam training model: {str(e)}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Kembali ke Step 3"):
                st.session_state.training_step = 3
                st.rerun()
    else:
        st.warning("⚠️ Tidak ada data yang tersedia untuk training.")

def show_evaluation_step():
    st.subheader("📈 Step 5: Evaluation & Model Saving")
    
    if 'current_model' in st.session_state:
        model_data = st.session_state.current_model
        
        # Show data information
        st.write("**📊 Informasi Data dari Multiple Files:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Data", model_data.get('total_data', 0))
        
        with col2:
            train_size = len(model_data['train_data'][0]) if 'train_data' in model_data else 0
            st.metric("Data Training", train_size)
        
        with col3:
            test_size = len(model_data['test_data'][0]) if 'test_data' in model_data else 0
            st.metric("Data Testing", test_size)
        
        with col4:
            source_files = model_data.get('source_files', {})
            total_files = source_files.get('rainfall_files', 0) + source_files.get('radiosonde_files', 0)
            st.metric("Total Files", total_files)
        
        # Show source files information
        if 'source_files' in model_data:
            st.write("**📁 Informasi File Sumber:**")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"• File Curah Hujan: {source_files.get('rainfall_files', 0)}")
            with col2:
                st.write(f"• File Radiosonde: {source_files.get('radiosonde_files', 0)}")
        
        # Show SMOTE information if used
        if model_data.get('use_smote', False):
            st.write("**⚖️ Informasi SMOTE Oversampling:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Distribusi Sebelum SMOTE:**")
                original_dist = model_data.get('original_distribution', {})
                for label, count in original_dist.items():
                    st.write(f"• {label}: {count} sampel")
            
            with col2:
                st.write("**Distribusi Setelah SMOTE:**")
                resampled_dist = model_data.get('resampled_distribution', {})
                for label, count in resampled_dist.items():
                    st.write(f"• {label}: {count} sampel")
        
        # Show hyperparameter tuning information if used
        if model_data.get('hyperparameter_tuning', {}).get('used', False):
            st.write("**🔧 Informasi Hyperparameter Tuning:**")
            
            tuning_info = model_data['hyperparameter_tuning']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Metode**: {tuning_info['method']}")
                st.write(f"**Scoring Metric**: {tuning_info['scoring']}")
                st.write("**Parameter Terbaik:**")
                for param, value in tuning_info['best_params'].items():
                    st.write(f"• **{param}**: {value}")
            
            with col2:
                # Show cross-validation results summary
                cv_results = tuning_info.get('cv_results', {})
                if cv_results:
                    mean_scores = cv_results.get('mean_test_score', [])
                    if len(mean_scores) > 0:
                        st.write("**Cross-Validation Results:**")
                        st.write(f"• **Best Score**: {max(mean_scores):.4f}")
                        st.write(f"• **Mean Score**: {np.mean(mean_scores):.4f}")
                        st.write(f"• **Std Score**: {np.std(mean_scores):.4f}")
                        st.write(f"• **Total Combinations**: {len(mean_scores)}")
                
                # Show improvement over default parameters
                st.write("**🎯 Optimasi Berhasil!**")
                st.write("Parameter telah dioptimalkan untuk performa terbaik")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Metrik Evaluasi:**")
            st.write(f"**Algoritma:** {model_data['algorithm']}")
            st.write(f"**Akurasi:** {model_data['accuracy']:.2%}")
            
            # Show SMOTE information
            if model_data.get('use_smote', False):
                st.write("**SMOTE:** ✅ Diterapkan")
                smote_params = model_data.get('smote_params', {})
                if smote_params:
                    st.write(f"**K-Neighbors:** {smote_params.get('k_neighbors', 5)}")
            else:
                st.write("**SMOTE:** ❌ Tidak diterapkan")
            
            # Show hyperparameter tuning information
            if model_data.get('hyperparameter_tuning', {}).get('used', False):
                st.write("**Hyperparameter Tuning:** ✅ Diterapkan")
                tuning_method = model_data['hyperparameter_tuning']['method']
                st.write(f"**Metode Tuning:** {tuning_method}")
            else:
                st.write("**Hyperparameter Tuning:** ❌ Tidak diterapkan")
            
            st.write("**📋 Classification Report:**")
            st.text(model_data['report'])
        
        with col2:
            st.write("**🔥 Confusion Matrix:**")
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Get original class labels for display
            label_encoder = model_data.get('label_encoder')
            if label_encoder is not None:
                try:
                    display_labels = label_encoder.inverse_transform(model_data['classes'])
                except:
                    display_labels = model_data['classes']
            else:
                display_labels = model_data['classes']
            
            sns.heatmap(
                model_data['confusion_matrix'], 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=display_labels,
                yticklabels=display_labels,
                ax=ax
            )
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title('Confusion Matrix')
            
            st.pyplot(fig)
        
        st.markdown("---")
        st.write("**💾 Simpan Model dari Multiple Dataset:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_name = st.text_input("Nama Model", value=f"{model_data['algorithm']}_MultiFile_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        with col2:
            if st.button("💾 Simpan ke Database", type="primary"):
                st.session_state.models[model_name] = {
                    'model': model_data['model'],
                    'algorithm': model_data['algorithm'],
                    'labels': model_data['labels'],
                    'label_encoder': model_data.get('label_encoder'),
                    'metrics': {
                        'accuracy': model_data['accuracy'],
                        'report': model_data['report'],
                        'confusion_matrix': model_data['confusion_matrix'],
                        'use_smote': model_data.get('use_smote', False),
                        'smote_params': model_data.get('smote_params', {}),
                        'total_data': model_data.get('total_data', 0),
                        'original_distribution': model_data.get('original_distribution', {}),
                        'resampled_distribution': model_data.get('resampled_distribution', {}),
                        'source_files': model_data.get('source_files', {})
                    },
                    'created_date': datetime.now()
                }
                st.success(f"✅ Model '{model_name}' berhasil disimpan!")
        
        with col3:
            # Save model with label encoder
            model_package = {
                'model': model_data['model'],
                'label_encoder': model_data.get('label_encoder'),
                'labels': model_data['labels'],
                'algorithm': model_data['algorithm'],
                'source_files': model_data.get('source_files', {}),
                'created_date': datetime.now()
            }
            model_pickle = pickle.dumps(model_package)
            st.download_button(
                label="📥 Download Model (.pkl)",
                data=model_pickle,
                file_name=f"{model_name}.pkl",
                mime="application/octet-stream"
            )
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Train Model Baru"):
                st.session_state.training_step = 4
                st.rerun()
        
        with col2:
            if st.button("🏠 Kembali ke Home"):
                st.session_state.current_page = "home"
                st.rerun()
        
        with col3:
            if st.button("🔮 Mulai Prediksi"):
                st.session_state.current_page = "prediction"
                st.rerun()
    
    else:
        st.warning("⚠️ Tidak ada model yang tersedia untuk evaluasi. Silakan kembali ke Step 4 untuk training model.")
        if st.button("← Kembali ke Step 4"):
            st.session_state.training_step = 4
            st.rerun()

def main():
    initialize_session_state()
    
    if st.session_state.current_page == "home":
        show_home()
    elif st.session_state.current_page == "prediction":
        show_prediction_system()
    elif st.session_state.current_page == "upload_model":
        show_upload_model_system()
    elif st.session_state.current_page == "training":
        show_training_system()

if __name__ == "__main__":
    main()
