## Thay đổi gì?

<!-- 1 đến 3 dòng: phần nào (A, B hay C), file nào, vì sao -->

## Checklist trước khi nhờ review

Đánh dấu `[x]` khi đã kiểm tra. Mục nào không áp dụng thì ghi "N/A".

**Hợp đồng giao diện (`src/interfaces.py`)**

- [ ] Preprocessor chỉ `fit` trên tập train (không dùng val, test)
- [ ] Đổi `train_size` không làm thay đổi tập val và test
- [ ] Mô hình có `preprocess_kind`, `n_params > 0` sau khi fit, `device = "cuda"` nếu train trên GPU
- [ ] `predict` trả xác suất lớp 1 trong [0, 1] (phân loại) hoặc giá trị thực (hồi quy)
- [ ] Early stopping (nếu có) chỉ dùng tập val, không dùng test

**So sánh công bằng**

- [ ] Search space ở `configs/search_spaces/<model>.yaml`, tuning dùng đúng `n_trials` chung
- [ ] Seed truyền vào mô hình (kết quả lặp lại được khi chạy lại cùng seed)
- [ ] Không sửa file của phần khác mà không báo trong mô tả PR

**Kiểm thử**

- [ ] `pytest -q` pass (test của phần mình không còn bị skip)
- [ ] Smoke chạy được: `python scripts/run_full.py --smoke` và `python scripts/make_report.py --smoke`
- [ ] Không commit `results/smoke/`, cache hay file dữ liệu tải về

**Tài liệu**

- [ ] README cập nhật nếu thêm lệnh, tham số hoặc file cấu hình
- [ ] Khai báo AI (README, mục khai báo AI) nếu có dùng công cụ AI
