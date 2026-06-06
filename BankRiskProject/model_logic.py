import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import joblib

# =================================================================================================
# هذا الملف يحتوي على منطق الذكاء الاصطناعي (Model Logic)
# الهدف: بناء وتدريب نموذج Random Forest لتصنيف المخاطر
# تحديث: إضافة مقاييس الأداء الأكاديمية (Confusion Matrix, F1 Score)
# =================================================================================================

class RiskModel:
    def __init__(self):
        """
        تهيئة النموذج.
        سنستخدم Random Forest لأنه قوي في التعامل مع البيانات المجدولة
        ويعطينا أهمية الخصائص (Feature Importance) للشرح.
        """
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        self.metrics = {} # لتخزين نتائج الأداء
        
    def preprocess(self, df):
        """
        تجهيز البيانات للتدريب:
        - حذف الأعمدة غير المؤثرة (مثل المعرف)
        - تحويل الهدف (Low, Medium, High) إلى أرقام
        """
        # نسخ البيانات لتجنب تعديل الأصل
        data = df.copy()
        
        # حذف معرف العميل لأنه لا يحمل قيمة تنبؤية
        if 'Customer_ID' in data.columns:
            data = data.drop(columns=['Customer_ID'])
            
        # تحويل الهدف (Risk_Level) إلى أرقام إذا كان موجوداً (للتدريب)
        # Low -> 0, Medium -> 1, High -> 2
        mapping = {'Low': 0, 'Medium': 1, 'High': 2}
        
        if 'Risk_Level' in data.columns:
            # التحقق من أن القيم صحيحة
            data['Risk_Level'] = data['Risk_Level'].map(mapping)
            
        return data

    def train(self, df):
        """
        تدريب النموذج على البيانات المقدمة.
        """
        processed_df = self.preprocess(df)
        
        # فصل الميزات (X) عن الهدف (y)
        X = processed_df.drop(columns=['Risk_Level'])
        y = processed_df['Risk_Level']
        
        # تقسيم البيانات إلى تدريب واختبار (80% تريب - 20% اختبار)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # تدريب النموذج
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.feature_names = list(X.columns) # حفظ ترتيب الأعمدة المستخدم في التدريب
        
        # تقييم سريع للدقة
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        
        # حفظ المقاييس للاستخدام في الواجهة
        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
        
        print(f"تم تدريب النموذج بنجاح. دقة النموذج: {accuracy * 100:.2f}%")
        return accuracy

    def predict_risk(self, input_data):
        """
        توقع مستوى المخاطر لبيانات عميل جديد.
        :param input_data: قاموس يحتوي على بيانات العميل
        :return: (Risk Score Class, Probability)
        """
        if not self.is_trained:
            raise Exception("يجب تدريب النموذج أولاً!")
            
        # تحويل القاموس إلى DataFrame (صف واحد)
        input_df = pd.DataFrame([input_data])
        
        # إعادة ترتيب الأعمدة لتطابق الترتيب المستخدم في التدريب (حل مشكلة ValueError)
        input_df = input_df[self.feature_names]
        
        prediction = self.model.predict(input_df)[0]
        probs = self.model.predict_proba(input_df)[0]
        
        # إرجاع النتيجة
        # 0->Low, 1->Medium, 2->High
        levels = {0: 'Low', 1: 'Medium', 2: 'High'}
        risk_level = levels[prediction]
        
        # درجة المخاطر كنسبة مئوية (بناءً على احتمال المخاطر العالية والمتوسطة)
        # معادلة تقريبية: (احتمال High * 100) + (احتمال Medium * 50)
        calculated_score = (probs[2] * 100) + (probs[1] * 50)
        risk_score = min(calculated_score, 100) # سقف 100
        
        return risk_level, risk_score, probs

    def get_feature_importance(self, feature_names):
        """
        استخراج أهم العوامل المؤثرة في القرار.
        """
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        sorted_features = []
        for i in range(len(importances)):
            sorted_features.append((feature_names[indices[i]], importances[indices[i]]))
            
        return sorted_features

    def get_performance_metrics(self):
        """إرجاع مقاييس الأداء لعرضها في الواجهة"""
        return self.metrics
