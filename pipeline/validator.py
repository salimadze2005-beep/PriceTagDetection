class Validator:
    def validate(self, data):
        # Сравнение штрихкодов, если есть оба
        barcode_ocr = data.get('barcode')
        barcode_qr = data.get('qr_code_barcode')
        if barcode_ocr and barcode_qr:
            data['barcode_conflict'] = (barcode_ocr != barcode_qr)
        else:
            data['barcode_conflict'] = None  # недостаточно данных
        return data