# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Nguyen Van Han

Công cụ tôi dùng: CVAT Docker trên máy, task 16, nhãn `car`; tôi xuất nhãn ở định dạng `Ultralytics YOLO Detection 1.0`.

## 1. Dữ liệu và cách chia tập

Tôi dùng cách chia theo thời gian của bộ dữ liệu: 268 ảnh pool để chọn ảnh gán nhãn, 20 ảnh test để chấm và 112 ảnh vùng đệm. Video quay liên tục từ một camera cố định, mỗi ảnh cách nhau 0,4 giây nên cùng một xe có thể nằm trong nhiều ảnh liền nhau. Nếu chia ngẫu nhiên, cùng xe và nền cảnh gần như giống nhau có thể xuất hiện ở cả train lẫn test, khiến điểm số cao hơn khả năng nhận xe mới trong thực tế. Vùng đệm giữ ảnh pool gần ảnh test nhất cách 4,4 giây (`data/DATA.md`), dù các ảnh vẫn thuộc cùng một camera và cùng một đêm.

## 2. Mô hình khởi đầu lạnh

Ở lần chạy đầu, tôi dùng `yolov8n` đã học từ COCO và gộp ba lớp car, bus, truck thành `car` của bài. Trên 20 ảnh test, mô hình đạt AP50 **0,7714**, P@0,25 **0,9249**, R@0,25 **0,4888**, F1 **0,6396** (`outputs/metrics_round0.json`). Khi xem `compare_round0.jpg`, tôi thấy mô hình hay bỏ sót xe xa, xe tối hoặc xe đứng sát nhau: `frame_0050` có 7 FN, `frame_0150` có 10 FN và `frame_0350` có 14 FN. Recall của xe small chỉ **0,1818** trên 66 box, thấp hơn medium **0,5473** trên 296 box và large **0,5610** trên 41 box.

Tôi không coi mọi chênh lệch trong ảnh so sánh là lỗi của mô hình. Ở `frame_0150`, cụm đèn hậu bên phải có các box tham chiếu gần nhau; tôi cần xem ảnh gốc để xác định đó là hai xe riêng hay nhãn bị chồng trước khi kết luận một dự đoán là FP. Nhãn test do một mô hình khác tạo và chưa được người rà từng box (`data/DATA.md`), nên tôi giữ nguyên chúng và ghi giới hạn này trong kết luận.

## 3. Chiến lược chọn mẫu

Tôi giữ chiến lược `uncertainty` của notebook và chọn 12 ảnh theo `score = 0,5 U + 0,3 A + 0,2 D`. `U` là độ bất định của các box khó nhất, `A` là số dự đoán mơ hồ đã chuẩn hóa, còn `D` là khoảng cách thời gian tới ảnh đã gán. Tôi giữ `MIN_GAP_S = 2` để hai ảnh trong cùng lô không quá sát nhau. Danh sách lô nằm trong `outputs/selection_round1.csv`; ở `reports/SELECTION.md` tôi giải thích cách ưu tiên nếu chỉ đủ công rà năm ảnh.

Tôi chú ý `frame_0182` đứng đầu (score 0,9591; U 0,9182; A 1,0) vì có nhiều xe xa, sát nhau; `frame_0369` đứng thứ hai (0,9324) vì cảnh đông; `frame_0099` ở hạng 8 (0,9063) cho tôi thêm một đoạn sớm và là ảnh tôi đã quét mù trước khi xem pre-label. Tôi cũng xem `frame_0372` (0,9101), nhưng ảnh này chỉ cách `0369` 1,2 giây nên không nằm cùng lô. Điểm cao giúp tôi ưu tiên công rà chứ không bảo đảm ảnh sẽ làm AP50 tăng; kết quả vòng 1 cho thấy điều đó.

## 4. Các vòng học chủ động

| Vòng | Model | Ảnh train | Box train | AP50 | Δ so với cold start | P@0,25 | R@0,25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n COCO cold start | 0 | 0 | 0,7714 | — | 0,9249 | 0,4888 | 0,6396 | 0,1818 | 0,5473 | 0,5610 |
| 1 | yolov8n fine-tune 50 epoch | 12 | 231 | 0,2895 | -0,4819 | 1,0000 | 0,0050 | 0,0099 | 0,0000 | 0,0000 | 0,0488 |

Nguồn: `reports/rounds_table.md` và `outputs/metrics_round0.json`, `outputs/metrics_round1.json`. Cả hai vòng được chấm trên cùng 20 ảnh test, IoU 0,5; AP50 quét theo confidence, còn P/R/F1 ở confidence 0,25. P vòng 1 bằng 1,0 chỉ vì có **2 TP và 0 FP** ở ngưỡng này, đồng thời bỏ sót **401/403** box; đó không phải kết quả tốt. `compare_round1.jpg` cho thấy `frame_0050` từ 11 TP/7 FN ở cold start thành 0 TP/18 FN sau fine-tune, `frame_0150` từ 10 TP/10 FN thành 0 TP/20 FN. Recall giảm ở cả ba nhóm kích thước, nặng nhất là small và medium về 0.

Trước khi sửa nhãn, bản quét độc lập `BLIND_SCAN.md` đã ghi khoảng 21 xe ở `frame_0099`, trong đó xe tối bị cắt mép và các xe nhỏ dưới cầu dễ sót. Sau khi import 169 pre-label vào CVAT, tôi rà cả 12 ảnh, giữ **168** box, chỉnh **0**, xóa **1** box gộp nhiều xe, thêm **63** box; bản cuối là **231** box (`outputs/round1_diff.md`). `REVIEW_LOG.csv` ghi các ca cụ thể: xe sáng bị bỏ sót ở `0099`, xe xa ở `0182`, box gộp sai ở `0312`, và xe cắt mép ở `0392`. Theo `GUIDELINE_LABEL.md`, xe bị cắt mép chỉ khoanh phần nhìn thấy, hai xe sát nhau tách box, không lấy vệt đèn phản chiếu trên đường.

Nhãn đã sửa là dữ liệu train, không phải kết quả mô hình sau train. Việc thêm nhiều box cũng không tự chứng minh chúng làm model tốt hơn. Với 12 ảnh của một camera và 50 epoch, dữ liệu còn ít; việc confidence ở vòng 1 thấp hơn rõ tại ngưỡng 0,25 là một **dấu hiệu cần kiểm tra** về huấn luyện và hiệu chuẩn điểm, chưa đủ để quy một nguyên nhân duy nhất. Ảnh so sánh và metric không cho phép kết luận lỗi nằm ở CVAT hay riêng một box nào.

## 5. Kết luận và giới hạn

Vòng 1 giảm **0,4819 AP50** so với cold start, đồng thời recall ở ngưỡng 0,25 từ **0,4888** xuống **0,0050**. Tôi dừng ở vòng 1 để kiểm tra thay vì gán vòng 2 theo điểm số hiện tại. File `selection_round2.csv` có 5/12 ảnh được chọn mà model không đề xuất box nào, nên chiến lược chọn mẫu vòng sau đang bị ảnh hưởng bởi mô hình yếu.

Trước khi train thêm, cần rà lại vài ảnh train và export CVAT so với ảnh gốc; xác nhận class id 0, tọa độ chuẩn hóa, box không gộp xe hoặc ôm vệt đèn; xem phân bố confidence và các dự đoán ở ngưỡng thấp trên train/test; rồi thử đối chứng ít epoch hơn hoặc learning rate thấp hơn trên cùng 20 ảnh test. Không dùng số mAP Ultralytics tính trên ảnh train làm kết quả báo cáo. Nếu pipeline hợp lệ nhưng model vẫn bỏ sót, hai ca đáng rà ở vòng sau là `frame_0022` (8,8 giây, 0 box gợi ý) và `frame_0207` (82,8 giây, 0 box gợi ý), sau khi xem ảnh gốc để xác nhận có xe. Cả hai có thể tốn công vẽ từ đầu; cần tránh lấy thêm frame lân cận dưới 2 giây vì cảnh dễ gần trùng (`selection_round2.csv`).

Kết luận chỉ là mức khớp với **20 ảnh test** và **403 box được chấm**; 14 box cao dưới 16 px bị bỏ qua. Nhãn tham chiếu do mô hình tạo và chưa được rà tay, nên sai số nhãn có thể làm AP50 và recall lệch so với chất lượng phát hiện xe thật. Chỉ một video, một camera và 12 ảnh train cũng chưa chứng minh khả năng áp dụng cho cảnh khác. Mọi số liệu ở đây là kết quả Colab thực chạy; không sửa nhãn test để nâng điểm.
