import pandas as pd
import numpy as np
import random

# =================================================================================================
# هذا الملف مسؤول عن توليد البيانات الوهمية (Synthetic Data)
# الهدق: إنشاء قاعدة بيانات لعملاء بنك افتراضي لتدريب نموذج الذكاء الاصطناعي
# =================================================================================================

def generate_synthetic_data(num_samples=1000):
    """
    تقوم هذه الدالة بتوليد بيانات وهمية للعملاء لغرض التدريب.
    :param num_samples: عدد العينات (العملاء) المطلوب توليدها
    :return: Dataframe يحتوي على البيانات
    """
    
    # 1. إعداد البذور العشوائية لضمان تكرار النتائج (Reproducibility)
    np.random.seed(42)
    random.seed(42)
    
    data = {
        # معرف العميل (وهمي)
        'Customer_ID': [f'CUST_{i}' for i in range(1000, 1000 + num_samples)],
        
        # العمر: يؤثر أحياناً على السلوك الرقمي (كبار السن قد يكونون أكثر عرضة للهندسة الاجتماعية)
        'Age': np.random.randint(18, 80, num_samples),
        
        # عدد مرات فشل تسجيل الدخول في الأسبوع الماضي (مؤشر قوي على محاولات اختراق)
        'Login_Failures': np.random.poisson(lam=1, size=num_samples),
        
        # حالة تحديث نظام التشغيل (1: محدث، 0: قديم). الأنظمة القديمة بها ثغرات.
        'OS_Updates_Status': np.random.choice([0, 1], size=num_samples, p=[0.3, 0.7]),
        
        # تفعيل المصادقة الثنائية (2FA) - عامل حماية قوي جداً
        'Two_Factor_Auth': np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6]),
        
        # عدد الأيام منذ تغيير كلمة المرور (رقم كبير = خطر)
        'Password_Change_Days': np.random.randint(1, 365, num_samples),
        
        # هل يستخدم العميل VPN؟ (قد يستخدم لإخفاء الموقع أو قد يكون مؤشراً مشبوهاً في سياقات معينة)
        'VPN_Usage': np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2]),
        
        # عدد الأجهزة المختلفة التي تم الدخول منها في آخر شهر
        'Device_Count': np.random.randint(1, 6, num_samples),
        
        # هل تم رصد برمجيات خبيثة (Malware) سابقاً على أجهزة العميل؟ (1: نعم، 0: لا)
        'Malware_History': np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1]),
        
        # متوسط وقيمة التحويلات المالية (مبالغ ضخمة مفاجئة قد تكون احتيالاً)
        'Avg_Transaction_Amt': np.random.normal(500, 200, num_samples),
        
        # عدد مرات تغيير الموقع الجغرافي (IP) في وقت قصير بشكل غير منطقي
        'Suspicious_Location_Changes': np.random.choice([0, 1, 2, 3], size=num_samples, p=[0.8, 0.15, 0.04, 0.01]),
        
        # النقرات على روابط تصيد (Phishing) في اختبارات التوعية السابقة
        'Phishing_Clicks': np.random.randint(0, 5, num_samples)
    }
    
    df = pd.DataFrame(data)
    
    # 2. حساب "عمود الهدف" (Target Label) بناءً على قواعد منطقية لتدريب النموذج
    # في الواقع، هذا العمود يأتي من بيانات تاريخية حقيقية (تم اختراقهم أم لا).
    # هنا سنصنعه بناءً على المدخلات لكي "يتعلم" النموذج العلاقة.
    
    def calculate_risk_level(row):
        score = 0
        # زيادة المخاطر بناءً على فشل الدخول
        score += row['Login_Failures'] * 10
        
        # الأنظمة غير المحدثة تزيد المخاطر
        if row['OS_Updates_Status'] == 0:
            score += 15
            
        # عدم وجود مصادقة ثنائية يزيد الخطر بشكل كبير
        if row['Two_Factor_Auth'] == 0:
            score += 20
            
        # كلمة مرور قديمة جداً
        if row['Password_Change_Days'] > 180:
            score += 10
            
        # وجود تاريخ برمجيات خبيثة خطر جداً
        if row['Malware_History'] == 1:
            score += 30
            
        # تغيير المواقع بشكل مريب
        score += row['Suspicious_Location_Changes'] * 20
        
        # كثرة الأجهزة
        if row['Device_Count'] > 3:
            score += 10
            
        # النقر على روابط التصيد
        score += row['Phishing_Clicks'] * 5
        
        # التصنيف النهائي
        if score < 20:
            return 'Low'      # مخاطر منخفضة
        elif score < 50:
            return 'Medium'   # مخاطر متوسطة
        else:
            return 'High'     # مخاطر عالية (خطر!!)

    df['Risk_Level'] = df.apply(calculate_risk_level, axis=1)
    
    # تحويل القيم الفئوية إلى أرقام إذا لزم الأمر في بعض النماذج، لكن سنتركها هكذا للعرض
    # سيقوم الموديل بالتعامل معها لاحقاً.
    
    return df

# اختبار سريع عند التشغيل المباشر
if __name__ == "__main__":
    df = generate_synthetic_data()
    print("تم توليد بيانات وهمية بنجاح.")
    print(df.head())
