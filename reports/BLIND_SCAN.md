# Quét độc lập trước khi xem pre-label

Frame: `frame_0099.jpg` (ảnh thuộc lô 12 ảnh được Colab chọn ở vòng 1).

Số xe nhìn thấy bằng mắt: khoảng 21 xe. Đây là đếm nhanh trên ảnh tối, có thể lệch ở các xe rất xa hoặc bị cắt ở mép ảnh.

Hai vị trí dễ bị AI bỏ sót hoặc vẽ sai:

1. Mép phải phía dưới: xe tối bị cắt một phần ở rìa khung hình, chỉ thấy thân xe và vùng sáng nhỏ; box cần ôm phần thân xe còn nhìn thấy, không ôm vùng phản chiếu.
2. Khu vực xa ngay dưới cầu và biển báo xanh (phía trên bên trái/giữa): nhiều xe nhỏ chỉ còn đèn đỏ hoặc đèn pha; dễ gộp hai xe, nhầm đèn với thân xe hoặc vẽ box quá lớn.

Quan sát này được ghi trước khi mở nhãn gợi ý của mô hình.
