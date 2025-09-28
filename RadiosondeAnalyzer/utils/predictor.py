import pandas as pd
import numpy as np
from datetime import datetime

class Predictor:
    def __init__(self):
        pass
    
    def prepare_features(self, data):
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        required_columns = ['KI', 'LI', 'SI', 'TT', 'CAPE']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0
        
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.fillna(0)
        
        features = df[required_columns]
        
        return features, df
    
    def predict(self, data, model, label_rules, class_mapping=None, label_encoder=None):
        features, df = self.prepare_features(data)
        
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        
        results = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            if 'Tanggal' in row and 'Jam' in row:
                date_str = f"{row['Tanggal']}-{row['Jam']}:00"
            else:
                date_str = f"Row {i+1}"
            
            prediction = predictions[i]
            prediction_index = list(model.classes_).index(prediction)
            probability = probabilities[i][prediction_index]
            
            # Convert prediction back to original string label if label encoder exists
            if label_encoder is not None:
                try:
                    display_prediction = label_encoder.inverse_transform([prediction])[0]
                except:
                    display_prediction = prediction
            else:
                display_prediction = prediction
            
            # Apply class mapping if available (for additional customization)
            if class_mapping and str(prediction) in class_mapping:
                display_prediction = class_mapping[str(prediction)]
            
            result = {
                'date': date_str,
                'KI': row['KI'],
                'LI': row['LI'],
                'SI': row['SI'],
                'TT': row['TT'],
                'CAPE': row['CAPE'],
                'prediction': display_prediction,
                'probability': probability
            }
            
            results.append(result)
        
        return results
    
    def predict_batch(self, data, model, label_rules, class_mapping=None, label_encoder=None):
        features, df = self.prepare_features(data)
        
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        
        results = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            prediction = predictions[i]
            prediction_index = list(model.classes_).index(prediction)
            probability = probabilities[i][prediction_index]
            
            # Convert prediction back to original string label if label encoder exists
            if label_encoder is not None:
                try:
                    display_prediction = label_encoder.inverse_transform([prediction])[0]
                except:
                    display_prediction = prediction
            else:
                display_prediction = prediction
            
            # Apply class mapping if available (for additional customization)
            if class_mapping and str(prediction) in class_mapping:
                display_prediction = class_mapping[str(prediction)]
            
            result = {
                'KI': row['KI'],
                'LI': row['LI'],
                'SI': row['SI'],
                'TT': row['TT'],
                'CAPE': row['CAPE'],
                'prediction': display_prediction,
                'probability': probability
            }
            
            results.append(result)
        
        return results
