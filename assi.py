import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt

class ImageProcessorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.original_image = None
        self.processed_image = None

    def init_ui(self):
        # إعدادات النافذة
        self.setWindowTitle("محول الصور الاحترافي - PyQt6 & OpenCV")
        self.setFixedSize(500, 450)
        
        # التصميم (Layout)
        layout = QVBoxLayout()

        self.label_title = QLabel("نظام معالجة الصور (24-bit to 8-bit)")
        self.label_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 20px;")
        layout.addWidget(self.label_title)

        # زر تحميل الصورة
        self.btn_load = QPushButton("1. إدخال صورة RGB")
        self.btn_load.setFixedHeight(40)
        self.btn_load.clicked.connect(self.load_image)
        layout.addWidget(self.btn_load)

        # زر التحويل
        self.btn_convert = QPushButton("2. تحويل إلى 8-bit Indexed")
        self.btn_convert.setFixedHeight(40)
        self.btn_convert.clicked.connect(self.convert_image)
        layout.addWidget(self.btn_convert)

        # زر الحفظ
        self.btn_save = QPushButton("3. حفظ الصورة على القرص")
        self.btn_save.setFixedHeight(40)
        self.btn_save.clicked.connect(self.save_image)
        layout.addWidget(self.btn_save)

        # منطقة عرض الحالة
        self.status_label = QLabel("الحالة: في انتظار تحميل صورة...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #555; margin-top: 20px;")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "اختر صورة", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            # استخدام OpenCV لقراءة الصورة
            self.original_image = cv2.imread(file_path)
            if self.original_image is not None:
                self.status_label.setText("تم تحميل الصورة بنجاح")
                QMessageBox.information(self, "نجاح", "تم استيراد بيانات الصورة (24-bit RGB)")
            else:
                QMessageBox.critical(self, "خطأ", "فشل في قراءة ملف الصورة")

    def convert_image(self):
        if self.original_image is None:
            QMessageBox.warning(self, "تنبيه", "يرجى تحميل صورة أولاً!")
            return

        # الخوارزمية اليدوية لتقليل الألوان (Color Quantization)
        # تحويل الصورة إلى RGB لأن OpenCV يقرأ بصيغة BGR
        img_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        # تقليل الألوان يدوياً عبر تقسيم النطاق (256 مستواً) إلى مستويات أقل
        # للوصول لـ 256 لوناً إجمالياً (8-bit):
        # سنستخدم 8 مستويات للأحمر (3 بت)، 8 مستويات للأخضر (3 بت)، و 4 مستويات للأزرق (2 بت)
        
        quantized = img_rgb.astype(np.float32)
        quantized[:, :, 0] = np.floor(quantized[:, :, 0] / 32) * 32  # Red
        quantized[:, :, 1] = np.floor(quantized[:, :, 1] / 32) * 32  # Green
        quantized[:, :, 2] = np.floor(quantized[:, :, 2] / 64) * 64  # Blue
        
        self.processed_image = quantized.astype(np.uint8)
        self.status_label.setText("الحالة: تم التحويل إلى 8-bit بنجاح")
        QMessageBox.information(self, "نجاح", "تمت عملية التحويل اللوني يدوياً")

    def save_image(self):
        if self.processed_image is None:
            QMessageBox.warning(self, "تنبيه", "لا توجد صورة معالجة لحفظها!")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "حفظ الصورة", "processed_image.png", "PNG Files (*.png);;BMP Files (*.bmp)")
        if file_path:
            # تحويل الصورة مرة أخرى إلى BGR قبل الحفظ لأن OpenCV يستخدم هذا التنسيق محلياً
            final_to_save = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, final_to_save)
            self.status_label.setText("الحالة: تم حفظ الملف بنجاح")
            QMessageBox.information(self, "تم الحفظ", f"تم حفظ الصورة في:\n{file_path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessorApp()
    window.show()
    sys.exit(app.exec())
