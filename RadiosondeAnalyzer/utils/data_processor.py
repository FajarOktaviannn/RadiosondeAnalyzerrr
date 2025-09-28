import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

class DataProcessor:
    def __init__(self):
        pass
    
    def extract_year_from_filename(self, filename):
        """Extract year from filename if available"""
        if filename:
            year_match = re.search(r'(\d{4})', filename)
            if year_match:
                return int(year_match.group(1))
        return datetime.now().year
    
    def process_multiple_radiosonde_files(self, uploaded_files):
        """Process multiple radiosonde files and combine them"""
        all_radiosonde_data = []
        file_info = []
        
        for uploaded_file in uploaded_files:
            try:
                # Read file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Extract year from filename
                year = self.extract_year_from_filename(uploaded_file.name)
                
                # Add year and filename info to dataframe
                df['SOURCE_FILE'] = uploaded_file.name
                df['EXTRACTED_YEAR'] = year
                
                all_radiosonde_data.append(df)
                file_info.append({
                    'filename': uploaded_file.name,
                    'year': year,
                    'rows': len(df),
                    'columns': len(df.columns)
                })
                
                print(f"Processed radiosonde file: {uploaded_file.name} (Year: {year}, Rows: {len(df)})")
                
            except Exception as e:
                print(f"Error processing radiosonde file {uploaded_file.name}: {str(e)}")
                continue
        
        if all_radiosonde_data:
            # Combine all radiosonde dataframes
            combined_radiosonde = pd.concat(all_radiosonde_data, ignore_index=True)
            return combined_radiosonde, file_info
        else:
            return None, []
    
    def process_multiple_rainfall_files(self, uploaded_files):
        """Process multiple rainfall files and combine them"""
        all_rainfall_data = []
        file_info = []
        
        for uploaded_file in uploaded_files:
            try:
                # Read file with different separators
                if uploaded_file.name.endswith('.csv'):
                    try:
                        df = pd.read_csv(uploaded_file, sep=';')
                    except:
                        try:
                            df = pd.read_csv(uploaded_file, sep=',')
                        except:
                            df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Add filename info to dataframe
                df['SOURCE_FILE'] = uploaded_file.name
                
                all_rainfall_data.append(df)
                file_info.append({
                    'filename': uploaded_file.name,
                    'rows': len(df),
                    'columns': len(df.columns)
                })
                
                print(f"Processed rainfall file: {uploaded_file.name} (Rows: {len(df)})")
                
            except Exception as e:
                print(f"Error processing rainfall file {uploaded_file.name}: {str(e)}")
                continue
        
        if all_rainfall_data:
            # Combine all rainfall dataframes
            combined_rainfall = pd.concat(all_rainfall_data, ignore_index=True)
            return combined_rainfall, file_info
        else:
            return None, []
    
    def merge_datasets(self, rainfall_data, radiosonde_data, radiosonde_filenames=None):
        """
        Merge radiosonde and rainfall datasets based on date and observation time (00:00 and 12:00)
        Now supports multiple files
        """
        rainfall_df = rainfall_data.copy()
        radiosonde_df = radiosonde_data.copy()
        
        print("Processing rainfall data...")
        
        # Process rainfall data
        if 'DATA' in rainfall_df.columns and 'TIMESTAMP' in rainfall_df.columns and '6H' in rainfall_df.columns:
            # Parse datetime from DATA and TIMESTAMP columns
            rainfall_df['DATETIME'] = pd.to_datetime(
                rainfall_df['DATA'] + ' ' + rainfall_df['TIMESTAMP'].str.split('.').str[0], 
                format='%d/%m/%Y %H:%M:%S', 
                errors='coerce'
            )
            
            # Clean and convert 6H column to numeric
            rainfall_df['6H'] = rainfall_df['6H'].replace('', np.nan)
            rainfall_df['6H'] = pd.to_numeric(rainfall_df['6H'], errors='coerce')
            rainfall_df['6H'] = rainfall_df['6H'].fillna(0)
            
            # Filter only 00:00 and 12:00 observations
            rainfall_filtered = rainfall_df[
                (rainfall_df['DATETIME'].dt.hour == 0) | 
                (rainfall_df['DATETIME'].dt.hour == 12)
            ].copy()
            
            if rainfall_filtered.empty:
                raise ValueError("No rainfall data found at 00:00 or 12:00 hours")
            
            # Extract date components
            rainfall_filtered['DAY'] = rainfall_filtered['DATETIME'].dt.day
            rainfall_filtered['MONTH'] = rainfall_filtered['DATETIME'].dt.month
            rainfall_filtered['YEAR'] = rainfall_filtered['DATETIME'].dt.year
            rainfall_filtered['OBS_HOUR'] = rainfall_filtered['DATETIME'].dt.hour.astype(str).str.zfill(2)
            
            print(f"Rainfall data processed: {len(rainfall_filtered)} records at 00:00 and 12:00")
            
        else:
            raise ValueError("Expected columns 'DATA', 'TIMESTAMP', and '6H' not found in rainfall data")
        
        print("Processing radiosonde data...")
        
        # Process radiosonde data
        if 'TANGGAL' in radiosonde_df.columns and 'JAM' in radiosonde_df.columns:
            # Convert TANGGAL and JAM to proper format
            radiosonde_df['DAY'] = pd.to_numeric(radiosonde_df['TANGGAL'], errors='coerce').astype(int)
            radiosonde_df['OBS_HOUR'] = radiosonde_df['JAM'].astype(str).str.zfill(2)
            
            # Handle month information
            if 'BULAN' in radiosonde_df.columns:
                month_map = {
                    'JANUARY': 1, 'FEBRUARY': 2, 'MARCH': 3, 'APRIL': 4,
                    'MAY': 5, 'JUNE': 6, 'JULY': 7, 'AUGUST': 8,
                    'SEPTEMBER': 9, 'OCTOBER': 10, 'NOVEMBER': 11, 'DECEMBER': 12
                }
                radiosonde_df['MONTH'] = radiosonde_df['BULAN'].str.upper().map(month_map)
                
                # Handle missing month mapping
                radiosonde_df['MONTH'] = radiosonde_df['MONTH'].fillna(1)  # Default to January if not found
            else:
                # If no BULAN column, try to infer from rainfall data or use current month
                if not rainfall_filtered.empty:
                    radiosonde_df['MONTH'] = rainfall_filtered['MONTH'].iloc[0]
                else:
                    radiosonde_df['MONTH'] = datetime.now().month
            
            # Handle year information - use EXTRACTED_YEAR if available
            if 'EXTRACTED_YEAR' in radiosonde_df.columns:
                radiosonde_df['YEAR'] = radiosonde_df['EXTRACTED_YEAR']
                print(f"Using extracted years from filenames")
            elif not rainfall_filtered.empty:
                # Use year from rainfall data
                radiosonde_df['YEAR'] = rainfall_filtered['YEAR'].iloc[0]
                print(f"Year taken from rainfall data: {rainfall_filtered['YEAR'].iloc[0]}")
            else:
                # Default to current year
                radiosonde_df['YEAR'] = datetime.now().year
                print(f"Using current year: {datetime.now().year}")
            
            # Create datetime for radiosonde data
            radiosonde_df['DATETIME'] = pd.to_datetime(
                radiosonde_df['YEAR'].astype(str) + '-' + 
                radiosonde_df['MONTH'].astype(str).str.zfill(2) + '-' + 
                radiosonde_df['DAY'].astype(str).str.zfill(2) + ' ' + 
                radiosonde_df['OBS_HOUR'] + ':00:00',
                errors='coerce'
            )
            
            # Clean numeric columns
            numeric_columns = ['KI', 'LI', 'SI', 'TT', 'CAPE']
            for col in numeric_columns:
                if col in radiosonde_df.columns:
                    radiosonde_df[col] = pd.to_numeric(radiosonde_df[col], errors='coerce')
            
            print(f"Radiosonde data processed: {len(radiosonde_df)} records")
            
        else:
            raise ValueError("Expected columns 'TANGGAL' and 'JAM' not found in radiosonde data")
        
        print("Merging datasets...")
        
        # Merge datasets based on date and observation hour
        merged_data = pd.merge(
            radiosonde_df[['DAY', 'MONTH', 'YEAR', 'OBS_HOUR', 'KI', 'LI', 'SI', 'TT', 'CAPE', 'DATETIME', 'SOURCE_FILE']],
            rainfall_filtered[['DAY', 'MONTH', 'YEAR', 'OBS_HOUR', '6H']],
            on=['DAY', 'MONTH', 'YEAR', 'OBS_HOUR'],
            how='inner'  # Only keep records that exist in both datasets
        )
        
        if merged_data.empty:
            print("Warning: No matching records found between datasets")
            print("Radiosonde date range:", radiosonde_df['DATETIME'].min(), "to", radiosonde_df['DATETIME'].max())
            print("Rainfall date range:", rainfall_filtered['DATETIME'].min(), "to", rainfall_filtered['DATETIME'].max())
            print("Radiosonde observation hours:", radiosonde_df['OBS_HOUR'].unique())
            print("Rainfall observation hours:", rainfall_filtered['OBS_HOUR'].unique())
            
            # Try a more flexible merge approach
            print("Attempting flexible merge...")
            
            # Create a more flexible matching key
            radiosonde_df['MATCH_KEY'] = (
                radiosonde_df['MONTH'].astype(str).str.zfill(2) + '-' +
                radiosonde_df['DAY'].astype(str).str.zfill(2) + '-' +
                radiosonde_df['OBS_HOUR']
            )
            
            rainfall_filtered['MATCH_KEY'] = (
                rainfall_filtered['MONTH'].astype(str).str.zfill(2) + '-' +
                rainfall_filtered['DAY'].astype(str).str.zfill(2) + '-' +
                rainfall_filtered['OBS_HOUR']
            )
            
            merged_data = pd.merge(
                radiosonde_df[['MATCH_KEY', 'KI', 'LI', 'SI', 'TT', 'CAPE', 'DATETIME', 'SOURCE_FILE']],
                rainfall_filtered[['MATCH_KEY', '6H']],
                on='MATCH_KEY',
                how='inner'
            )
        
        if merged_data.empty:
            raise ValueError("No matching records found between radiosonde and rainfall datasets. Please check date ranges and observation times.")
        
        # Format final output
        final_data = merged_data[['DATETIME', 'KI', 'LI', 'SI', 'TT', 'CAPE', '6H', 'SOURCE_FILE']].copy()
        
        # Sort by datetime
        final_data = final_data.sort_values('DATETIME').reset_index(drop=True)
        
        print(f"Successfully merged {len(final_data)} records")
        print("Date range:", final_data['DATETIME'].min(), "to", final_data['DATETIME'].max())
        
        # Show statistics by source file
        if 'SOURCE_FILE' in final_data.columns:
            print("\nData distribution by source file:")
            source_stats = final_data['SOURCE_FILE'].value_counts()
            for source, count in source_stats.items():
                print(f"  {source}: {count} records")
        
        return final_data
    
    def clean_data(self, data):
        """Clean the merged dataset"""
        cleaned_data = data.copy()
        
        # Remove rows where rainfall data (6H) is missing
        if '6H' in cleaned_data.columns:
            initial_count = len(cleaned_data)
            cleaned_data = cleaned_data.dropna(subset=['6H'])
            removed_count = initial_count - len(cleaned_data)
            if removed_count > 0:
                print(f"Removed {removed_count} rows with missing rainfall data")
        
        # Remove rows where any radiosonde feature is missing
        feature_columns = ['KI', 'LI', 'SI', 'TT', 'CAPE']
        for feature in feature_columns:
            if feature in cleaned_data.columns:
                initial_count = len(cleaned_data)
                cleaned_data = cleaned_data.dropna(subset=[feature])
                removed_count = initial_count - len(cleaned_data)
                if removed_count > 0:
                    print(f"Removed {removed_count} rows with missing {feature} data")
        
        # Ensure numeric types
        if '6H' in cleaned_data.columns:
            cleaned_data['6H'] = pd.to_numeric(cleaned_data['6H'], errors='coerce')
        
        for feature in feature_columns:
            if feature in cleaned_data.columns:
                cleaned_data[feature] = pd.to_numeric(cleaned_data[feature], errors='coerce')
        
        # Fill any remaining NaN values with 0
        cleaned_data = cleaned_data.fillna(0)
        
        print(f"Data cleaning completed. Final dataset: {len(cleaned_data)} records")
        
        return cleaned_data
    
    def apply_labels(self, data, target_column, label_rules):
        """Apply labeling rules to the target column"""
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        labeled_data = data.copy()
        labeled_data['label'] = None
        
        for rule in label_rules:
            min_val = rule['min']
            max_val = rule['max'] if rule['max'] != float('inf') else float('inf')
            label = rule['label']
            
            if max_val == float('inf'):
                mask = labeled_data[target_column] >= min_val
            else:
                mask = (labeled_data[target_column] >= min_val) & (labeled_data[target_column] < max_val)
            
            labeled_data.loc[mask, 'label'] = label
        
        # Remove rows without labels (shouldn't happen if rules cover all ranges)
        labeled_data = labeled_data.dropna(subset=['label'])
        
        return labeled_data
