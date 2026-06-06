import ttkbootstrap as ttk
from gui_app import BankRiskApp
import matplotlib.pyplot as plt

# =================================================================================================
# نقطة الدخول الرئيسية (Entry Point)
# =================================================================================================

def main():
    # استخدام ثيم "flatly" الاحترافي المناسب للبنوك
    # ثيمات مقترحة: cosmo, flatly, journal, lumen, minty, pulse, sandstone, united, yeti
    # ثيمات داكنة: cyborg, darkly, solar, superhero
    app_window = ttk.Window(themename="flatly")
    
    # محاولة تحسين العرض على شاشات عالية الدقة
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    
    # ضبط إعدادات الرسوم البيانية لتتوافق مع المظهر العام
    plt.style.use('ggplot')
    
    print("Application Starting...")
    print("Loading AI Models...")
    
    app = BankRiskApp(app_window)
    app_window.mainloop()

if __name__ == "__main__":
    main()
