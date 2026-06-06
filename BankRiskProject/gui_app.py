import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
import seaborn as sns
from data_generator import generate_synthetic_data
from model_logic import RiskModel
import datetime

# =================================================================================================
# هذا الملف مسؤول عن الواجهة الرسومية (GUI)
# تم تحديثه ليشمل:
# 1. الميزات الأمنية الجديدة (2FA, Password Age)
# 2. تبويب جديد لمقاييس الأداء الأكاديمية (Model Evaluation Dashboard)
# 3. إمكانية تصدير التقارير (Simulated PDF Export)
# =================================================================================================

class BankRiskApp:
    def __init__(self, root):
        self.root = root
        self.root.title("نظام التنبؤ بمخاطر العملاء البنكية - Cyber Risk AI")
        self.root.geometry("1500x900")
        
        # متغيرات الحالة
        self.status_var = tk.StringVar()
        self.status_var.set("جاري تهيئة النظام وتدريب النموذج...")
        
        # بناء التخطيط العام باستخدام Notebook (Tabbed Interface) لتنظيم أفضل
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=BOTH, expand=YES, padx=10, pady=10)
        
        # التبويب الرئيسي: التنبؤ (Prediction)
        self.prediction_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_tab, text="🔍 تحليل المخاطر (Prediction)")
        
        # التبويب الثاني: الأداء (Performance Metrics)
        self.metrics_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.metrics_tab, text="📊 أداء النموذج (Model Evaluation)")
        
        # بناء المحتويات
        self.create_prediction_layout()
        self.create_metrics_layout()
        
        # تدريب النموذج
        self.root.after(100, self.train_model)
        
    def train_model(self):
        """تدريب النموذج في الخلفية"""
        try:
            self.df = generate_synthetic_data(2000) # زيادة البيانات لدقة أفضل
            self.model = RiskModel()
            self.accuracy = self.model.train(self.df)
            
            self.status_var.set(f"النظام جاهز. دقة النموذج: {self.accuracy*100:.1f}%")
            self.plot_risk_distribution()
            self.plot_model_metrics() # تحديث رسوم الأداء أيضاً
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def create_prediction_layout(self):
        """تخطيط تبويب التنبؤ"""
        main_container = ttk.Frame(self.prediction_tab, padding=20)
        main_container.pack(fill=BOTH, expand=YES)
        
        # 1. لوحة المدخلات (Right Side)
        input_panel = ttk.Labelframe(main_container, text=" بيانات العميل (Customer Data) ", padding=15, bootstyle="info")
        input_panel.pack(side=RIGHT, fill=Y, padx=(10, 0))
        
        self.create_inputs(input_panel)
        
        # 2. لوحة النتائج والرسوم (Left Side)
        results_panel = ttk.Frame(main_container)
        results_panel.pack(side=LEFT, fill=BOTH, expand=YES, padx=(0, 10))
        
        # منطقة النتيجة الرقمية
        self.score_card = ttk.Labelframe(results_panel, text=" نتيجة التحليل (Analysis Result) ", padding=20, bootstyle="success")
        self.score_card.pack(fill=X, pady=(0, 20))
        
        self.result_label = ttk.Label(self.score_card, text="يرجى إدخال البيانات للبدء...", 
                                      font=("Helvetica", 16), bootstyle="secondary")
        self.result_label.pack(fill=X)
        
        # زر التصدير (مخفي مبدئياً)
        self.btn_export = ttk.Button(self.score_card, text="📄 تصدير تقرير (Export Report)", 
                                     command=self.export_report, state="disabled", bootstyle="outline-primary")
        self.btn_export.pack(anchor=NE, pady=5)
        
        # منطقة الرسوم البيانية
        self.charts_card = ttk.Labelframe(results_panel, text=" التحليلات البيانية (Live Analytics) ", padding=10, bootstyle="secondary")
        self.charts_card.pack(fill=BOTH, expand=YES)

    def create_metrics_layout(self):
        """تخطيط تبويب الأداء الأكاديمي"""
        # هذا التبويب لعرض الـ Confusion Matrix و F1 Score
        container = ttk.Frame(self.metrics_tab, padding=20)
        container.pack(fill=BOTH, expand=YES)
        
        header = ttk.Label(container, text="تقييم أداء نموذج الذكاء الاصطناعي (Model Performance Metrics)", font=("Helvetica", 18, "bold"))
        header.pack(pady=(0, 20))
        
        # حاوية للشارتات
        self.metrics_chart_frame = ttk.Frame(container)
        self.metrics_chart_frame.pack(fill=BOTH, expand=YES)
        
        # سيتم تعبئة هذا الإطار عند انتهاء التدريب في دالة plot_model_metrics

    def create_inputs(self, parent):
        """إنشاء حقول الإدخال مع التحديثات الجديدة"""
        
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=LEFT, fill=BOTH, expand=YES)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        self.vars = {}
        
        def add_input(label_text, var_name, kind='spin', min_v=0, max_v=100):
            container = ttk.Frame(scroll_frame, padding=(0, 5))
            container.pack(fill=X)
            
            lbl = ttk.Label(container, text=label_text, font=("Helvetica", 10, "bold"))
            lbl.pack(anchor=NE)
            
            if kind == 'binary':
                var = tk.IntVar(value=0)
                chk = ttk.Checkbutton(container, text="نعم / Yes", variable=var, bootstyle="round-toggle")
                chk.pack(anchor=NE, pady=2)
                self.vars[var_name] = var
                
            elif kind == 'spin':
                var = tk.DoubleVar(value=min_v) if isinstance(min_v, float) else tk.IntVar(value=min_v)
                spin = ttk.Spinbox(container, from_=min_v, to=max_v, textvariable=var)
                spin.pack(fill=X, pady=2)
                self.vars[var_name] = var

        # === User Info ===
        ttk.Label(scroll_frame, text="المعلومات الأساسية", font=("Helvetica", 12, "bold"), bootstyle="primary").pack(anchor=NE, pady=5)
        add_input("العمر (Age)", "Age", min_v=18, max_v=90)
        add_input("متوسط التحويلات ($)", "Avg_Transaction_Amt", min_v=0.0, max_v=50000.0)
        
        ttk.Separator(scroll_frame).pack(fill=X, pady=10)
        
        # === Behavioral Info ===
        ttk.Label(scroll_frame, text="السلوك الرقمي", font=("Helvetica", 12, "bold"), bootstyle="warning").pack(anchor=NE, pady=5)
        add_input("مرات فشل الدخول (Login Failures)", "Login_Failures", min_v=0, max_v=50)
        add_input("عدد الأجهزة (Device Count)", "Device_Count", min_v=1, max_v=20)
        add_input("تغيير الموقع المشبوه", "Suspicious_Location_Changes", min_v=0, max_v=50)
        add_input("نقرات التصيد (Phishing Clicks)", "Phishing_Clicks", min_v=0, max_v=20)
        
        ttk.Separator(scroll_frame).pack(fill=X, pady=10)
        
        # === Security Info (NEW FEATURES) ===
        ttk.Label(scroll_frame, text="الحالة الأمنية", font=("Helvetica", 12, "bold"), bootstyle="danger").pack(anchor=NE, pady=5)
        add_input("أيام منذ تغيير كلمة المرور", "Password_Change_Days", min_v=1, max_v=1000)
        add_input("نظام التشغيل محدث؟ (OS Updated)", "OS_Updates_Status", kind='binary')
        add_input("تفعيل المصادقة الثنائية (2FA)؟", "Two_Factor_Auth", kind='binary')
        add_input("استخدام VPN؟", "VPN_Usage", kind='binary')
        add_input("سجل برمجيات خبيثة؟ (Malware)", "Malware_History", kind='binary')
        
        ttk.Separator(scroll_frame).pack(fill=X, pady=20)
        
        # Buttons
        btn_analyze = ttk.Button(scroll_frame, text="⚡ تحليل المخاطر (Run Analysis)", 
                                 command=self.analyze_risk, bootstyle="danger")
        btn_analyze.pack(fill=X, pady=5, ipady=5)
        
        btn_random = ttk.Button(scroll_frame, text="🎲 بيانات عشوائية (Random Data)", 
                                command=self.fill_random_data, bootstyle="secondary-outline")
        btn_random.pack(fill=X, pady=5)
        
        lbl_status = ttk.Label(parent, textvariable=self.status_var, font=("Helvetica", 9), bootstyle="secondary")
        lbl_status.pack(side=BOTTOM, pady=5)

    def fill_random_data(self):
        """تعبئة عشوائية"""
        self.vars['Age'].set(np.random.randint(20, 70))
        self.vars['Login_Failures'].set(np.random.choice([0, 1, 3, 10]))
        self.vars['Device_Count'].set(np.random.randint(1, 5))
        self.vars['Avg_Transaction_Amt'].set(round(np.random.uniform(500, 5000), 2))
        self.vars['Suspicious_Location_Changes'].set(np.random.choice([0, 0, 1]))
        self.vars['Phishing_Clicks'].set(np.random.choice([0, 0, 1]))
        self.vars['Password_Change_Days'].set(np.random.randint(30, 400))
        
        self.vars['OS_Updates_Status'].set(np.random.choice([0, 1]))
        self.vars['Two_Factor_Auth'].set(np.random.choice([0, 1]))
        self.vars['VPN_Usage'].set(np.random.choice([0, 1]))
        self.vars['Malware_History'].set(np.random.choice([0, 0, 1]))

    def analyze_risk(self):
        if not hasattr(self, 'model'): return
        
        data = {k: v.get() for k, v in self.vars.items()}
        
        level, score, probs = self.model.predict_risk(data)
        
        self.current_result = {'level': level, 'score': score, 'probs': probs, 'data': data}
        
        self.display_result_card(level, score)
        self.update_charts(data, level)
        self.btn_export.config(state="normal") # تمكين زر التقرير

    def display_result_card(self, level, score):
        if level == "Low":
            style = "success"
            icon = "✅"
            msg = "العميل آمن (Low Risk). يوصى بالموافقة."
        elif level == "Medium":
            style = "warning"
            icon = "⚠️"
            msg = "مخاطر متوسطة. يوصى بالمراجعة واتصال التحقق."
        else:
            style = "danger"
            icon = "⛔"
            msg = "مخاطر عالية! يوصى بالرفض وتجميد الحساب."
            
        text = f"{icon} مستوى المخاطر: {level}\n🎯 درجة الخطر: {score:.1f}/100\n📝 التوصية: {msg}"
        
        self.result_label.config(text=text, bootstyle=style)
        self.score_card.config(bootstyle=style)

    def export_report(self):
        """تصدير تقرير نصي بسيط يحاكي PDF"""
        if not hasattr(self, 'current_result'): return
        
        res = self.current_result
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_text = f"""
        ====================================================
           BANK CUSTOMER RISK INTELLIGENCE REPORT
        ====================================================
        Date: {now}
        Customer ID: SIMULATED-ID-{np.random.randint(1000,9999)}
        ----------------------------------------------------
        RISK ASSESSMENT:
        Risk Level : {res['level']}
        Risk Score : {res['score']:.2f} / 100
        ----------------------------------------------------
        FACTOR ANALYSIS:
        - 2FA Enabled: {'Yes' if res['data']['Two_Factor_Auth'] else 'NO (Critical Risk)'}
        - OS Updated: {'Yes' if res['data']['OS_Updates_Status'] else 'NO'}
        - Login Failures: {res['data']['Login_Failures']}
        - Malware History: {'YES' if res['data']['Malware_History'] else 'No'}
        ----------------------------------------------------
        RECOMMENDATION:
        The system recommends evaluating this profile based on 
        the calculated risk probability of {res['probs'][2]*100:.1f}%.
        ====================================================
        """
        
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", 
                                                 filetypes=[("Text Report", "*.txt")],
                                                 initialfile=f"Risk_Report_{res['level']}.txt")
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            messagebox.showinfo("Export", "Report saved successfully.")

    def plot_risk_distribution(self):
        # الرسم في تبويب التنبؤ
        pass # سنقوم بتحديثه في update_charts 

    def update_charts(self, input_data, risk_level):
        for w in self.charts_card.winfo_children(): w.destroy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # 1. Feature Importance
        feats = self.model.get_feature_importance(list(input_data.keys()))
        top_feats = feats[:5]
        names = [x[0] for x in top_feats]
        vals = [x[1] for x in top_feats]
        
        sns.barplot(x=vals, y=names, ax=ax1, palette="mako")
        ax1.set_title("Top Risk Contributors")
        ax1.set_xlabel("Impact Score")
        
        # 2. Probability Donut Chart
        probs = self.current_result['probs']
        labels = ['Low', 'Medium', 'High']
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        ax2.pie(probs, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90, wedgeprops=dict(width=0.3))
        ax2.set_title("Probability Distribution")
        
        self.draw_figure(fig, self.charts_card)

    def plot_model_metrics(self):
        """رسم مقاييس الأداء في التبويب الثاني"""
        for w in self.metrics_chart_frame.winfo_children(): w.destroy()
        
        metrics = self.model.get_performance_metrics()
        if not metrics: return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. Confusion Matrix Heatmap
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                    xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
        ax1.set_title("Confusion Matrix (Accuracy Visualization)")
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Actual")
        
        # 2. Key Metrics Bar Chart
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
        
        sns.barplot(x=metric_names, y=metric_values, ax=ax2, palette='viridis', hue=metric_names, legend=False)
        ax2.set_ylim(0, 1.1)
        for i, v in enumerate(metric_values):
            ax2.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
        ax2.set_title("Model Performance Metrics")
        
        self.draw_figure(fig, self.metrics_chart_frame)

    def draw_figure(self, fig, parent):
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=YES, padx=5, pady=5)
#       plt.close(fig) # Avoid closing here to keep interactive if needed, but for TK usually safe to keep open or handle carefully.
